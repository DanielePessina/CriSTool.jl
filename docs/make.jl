using Documenter
using DocumenterVitepress
using CriSTool

DocMeta.setdocmeta!(CriSTool,
                    :DocTestSetup,
                    :(using CriSTool);
                    recursive = true)

makedocs(
    source = "src",
    build = "build",
    modules = [CriSTool],
    sitename = "CriSTool.jl",
    authors = "Daniele Pessina, Jerry Y. Y. Heng, and Maria M. Papathanasiou",
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/DanielePessina/CriSTool.jl",
        devbranch = "main",
        devurl = "dev"),
    pages = [
        "Home" => "index.md",
        "Guides" => [
            "Running simulations" => "simulation.md",
            "Solvers" => "solvers.md",
            "Kinetics" => "kinetics.md",
            "Measurements" => "measurements.md",
            "Temperature profiles" => "temperature-profiles.md",
            "Solubility models" => "saturation-models.md",
            "Parameter estimation" => "parameter-estimation.md",
            "Optimisation" => "optimisation.md",
            "ABCDE" => "abcde.md",
            "Sensitivity analysis" => "sensitivity.md",
            "Ensembles and uncertainty" => "uq-ensembles.md",
            "Bringing your own system" => "bring-your-own-system.md",
        ],
        "Tutorials" => "tutorials.md",
        "API reference" => [
            "Overview" => "api.md",
            "Simulation and results" => "api/simulation.md",
            "Process conditions" => "api/conditions.md",
            "Kinetic models and rates" => "api/kinetics.md",
            "Solvers and quadrature" => "api/solvers.md",
            "Measurements and losses" => "api/measurements.md",
            "Inference and uncertainty" => "api/inference.md",
            "Plotting and diagnostics" => "api/plotting.md",
        ],
    ],
    checkdocs = :exports,
    checkdocs_ignored_modules = [CriSTool.KissABC],
    linkcheck_ignore = [r"^https://github\.com/DanielePessina/CriSTool.jl/blob/main/examples/"],
    linkcheck = true,
)

if get(ENV, "DEPLOY_DOCS", "false") == "true"
    DocumenterVitepress.deploydocs(
        repo = "github.com/DanielePessina/CriSTool.jl",
        target = joinpath(@__DIR__, "build"),
        branch = "gh-pages",
        devbranch = "main",
        push_preview = true,
    )
end
