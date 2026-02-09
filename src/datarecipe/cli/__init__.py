"""Command-line interface for DataRecipe."""

import click

from datarecipe.cli._helpers import (  # noqa: F401
    console,
    display_recipe,
    recipe_to_markdown,
    validate_output_path,
)


# Main CLI group
@click.group()
@click.version_option(version="0.4.0", prog_name="datarecipe")
def main():
    """DataRecipe - Analyze AI dataset ingredients, estimate costs, and generate workflows."""
    pass


# --- Register commands from submodules ---

# analyze.py
from datarecipe.cli.analyze import (  # noqa: E402
    analyze,
    deep_guide,
    export,
    guide,
    list_sources,
    show,
)

main.add_command(analyze)
main.add_command(show)
main.add_command(export)
main.add_command(list_sources)
main.add_command(guide)
main.add_command(deep_guide)

# tools.py
from datarecipe.cli.tools import (  # noqa: E402
    allocate,
    batch,
    compare,
    cost,
    create,
    deploy,
    detect_strategy,
    enhanced_guide,
    extract_prompts,
    extract_rubrics,
    generate,
    ira,
    pii,
    profile,
    providers,
    quality,
    workflow,
)

main.add_command(create)
main.add_command(cost)
main.add_command(quality)
main.add_command(batch)
main.add_command(compare)
main.add_command(profile)
main.add_command(deploy)
main.add_command(providers)
main.add_command(workflow)
main.add_command(extract_rubrics)
main.add_command(extract_prompts)
main.add_command(detect_strategy)
main.add_command(allocate)
main.add_command(enhanced_guide)
main.add_command(generate)
main.add_command(pii)
main.add_command(ira)

# deep.py
from datarecipe.cli.deep import deep_analyze  # noqa: E402

main.add_command(deep_analyze)

# batch.py
from datarecipe.cli.batch import batch_from_radar, integrate_report  # noqa: E402

main.add_command(batch_from_radar)
main.add_command(integrate_report)

# infra.py
from datarecipe.cli.infra import cache_cmd, knowledge_cmd, watch_cmd  # noqa: E402

main.add_command(watch_cmd)
main.add_command(cache_cmd)
main.add_command(knowledge_cmd)

# spec.py
from datarecipe.cli.spec import analyze_spec  # noqa: E402

main.add_command(analyze_spec)
