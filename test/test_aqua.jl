# Aqua package-quality checks (Aqua.jl). Installed as a test-only dependency.
using Aqua

@testset "Aqua (package quality)" begin
    # Development tools such as Revise are intentionally kept outside the
    # package dependency graph (the no-breadcrumbs rule): the dev REPL loads
    # Revise from the user's development environment via startup.jl, and the
    # runtime Project.toml has no Revise entry. Aqua's stale-deps check has
    # nothing to ignore, so no exclusion is needed here.
    #
    # persistent_tasks is disabled: its probe precompiles the package in a
    # subprocess and waits for a done.log that Julia 1.12.4's precompile path
    # does not always emit ("done.log was not created, but precompilation
    # exited"); the failure is in the check's harness, not a task leak (the
    # package spawns no tasks at load).
    Aqua.test_all(CriSTool; persistent_tasks = false)
end
