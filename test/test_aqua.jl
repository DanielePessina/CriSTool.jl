# Aqua package-quality checks (Aqua.jl). Installed as a test-only dependency.
using Aqua

@testset "Aqua (package quality)" begin
    # Revise sits in [deps] as a dev-tool dependency: the dev REPL loads it via
    # ~/.julia/config/startup.jl and the global Julia env has no Revise, so it
    # cannot be dropped without rewiring that workflow. It is never `using`-ed
    # from src/, which is exactly why Aqua's stale-deps check would flag it —
    # ignore is the documented, reason-carrying exclusion.
    #
    # persistent_tasks is disabled: its probe precompiles the package in a
    # subprocess and waits for a done.log that Julia 1.12.4's precompile path
    # does not always emit ("done.log was not created, but precompilation
    # exited"); the failure is in the check's harness, not a task leak (the
    # package spawns no tasks at load).
    Aqua.test_all(CriSTool; stale_deps = (ignore = [:Revise],),
                  persistent_tasks = false)
end