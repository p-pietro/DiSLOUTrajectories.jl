using Test
using TestReports
using TestReports.EzXML: root, findall

function with_junit(run_tests, output)
    # Nest the reporting testset: TestReports 1.4's top-level display mutates
    # DefaultTestSet fields that are const on Julia 1.13.
    results = @static if VERSION >= v"1.13"
        Test.@with_testset ReportingTestSet("JUnit") run_tests()
    else
        Test.push_testset(ReportingTestSet("JUnit"))
        try
            run_tests()
        finally
            Test.pop_testset()
        end
    end
    flattened = TestReports.flatten_results!(results)
    document = report(flattened)
    # TestReports 1.4 emits <skip>; JUnit consumers expect <skipped>.
    for skipped in findall("//testcase/skip", root(document))
        skipped.name = "skipped"
    end
    open(output, "w") do io
        print(io, document)
    end
    summary = root(document)
    println("Test results: $(summary["tests"]) tests, $(summary["failures"]) failures, $(summary["errors"]) errors. Report: $output")
    for problem in findall("//testcase/failure | //testcase/error", summary)
        println(stderr, problem.content)
    end
    # ReportingTestSet records failures without throwing; preserve Pkg.test's status.
    any_problems(flattened) && error("Tests failed; see $output")
    return nothing
end
