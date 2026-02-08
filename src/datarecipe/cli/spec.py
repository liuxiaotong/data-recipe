"""Specification document analysis command."""

import json
import sys
from pathlib import Path

import click
from rich.console import Console

from datarecipe.cli._helpers import console


@click.command("analyze-spec")
@click.argument("file_path", type=click.Path(exists=True), required=False)
@click.option("--output-dir", "-o", default="./projects", help="Output directory")
@click.option(
    "--size", "-s", default=100, type=int, help="Target dataset size (for cost estimation)"
)
@click.option("--region", "-r", default="china", help="Region for cost calculation (china/us)")
@click.option(
    "--provider",
    "-p",
    default="anthropic",
    type=click.Choice(["anthropic", "openai"]),
    help="LLM provider",
)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Interactive mode: output prompt, wait for JSON input from stdin",
)
@click.option(
    "--from-json",
    "from_json",
    type=click.Path(exists=True),
    help="Load analysis from JSON file instead of using LLM",
)
def analyze_spec(
    file_path: str,
    output_dir: str,
    size: int,
    region: str,
    provider: str,
    interactive: bool,
    from_json: str,
):
    """
    Analyze a specification/requirements document and generate project artifacts.

    Supports PDF, Word (docx), images (png/jpg), and text files.
    Uses LLM to extract structured information and generate:
    - Annotation specification
    - Executive summary
    - Milestone plan
    - Cost breakdown
    - Industry benchmark comparison

    Three modes of operation:

    \b
    1. API mode (default): Uses LLM API to analyze document
       datarecipe analyze-spec requirements.pdf

    \b
    2. Interactive mode: For use within Claude Code/Desktop
       datarecipe analyze-spec requirements.pdf --interactive
       (Outputs prompt, waits for JSON on stdin)

    \b
    3. From JSON: Load pre-computed analysis
       datarecipe analyze-spec requirements.pdf --from-json analysis.json
    """
    import os

    from datarecipe.analyzers.spec_analyzer import SpecAnalyzer
    from datarecipe.generators.spec_output import SpecOutputGenerator

    # Validate arguments
    if not file_path and not from_json:
        console.print("[red]错误: 需要提供文档路径或 --from-json 参数[/red]")
        return

    # Display header (to stderr in interactive mode)
    output = console if not interactive else Console(file=sys.stderr)

    output.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    output.print("[bold cyan]  DataRecipe 需求文档分析[/bold cyan]")
    output.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")

    if file_path:
        file_name = Path(file_path).name
        output.print(f"文档: [bold]{file_name}[/bold]")
    output.print(f"目标规模: [bold]{size}[/bold] 条")
    output.print(f"区域: [bold]{region}[/bold]")

    if interactive:
        output.print("模式: [bold]交互模式[/bold] (等待 stdin 输入)\n")
    elif from_json:
        output.print("模式: [bold]从 JSON 加载[/bold]\n")
    else:
        output.print(f"LLM: [bold]{provider}[/bold]\n")

    try:
        analyzer = SpecAnalyzer(provider=provider)
        analysis = None

        # Mode 1: From JSON file
        if from_json:
            output.print("[dim]📄 从 JSON 加载分析结果...[/dim]")
            with open(from_json, encoding="utf-8") as f:
                extracted = json.load(f)

            # Parse document if provided (for metadata)
            doc = None
            if file_path:
                doc = analyzer.parse_document(file_path)
                if doc.has_images():
                    output.print(f"[green]✓ 文档解析完成 (包含 {len(doc.images)} 张图片)[/green]")
                else:
                    output.print("[green]✓ 文档解析完成[/green]")

            analysis = analyzer.create_analysis_from_json(extracted, doc)
            output.print(f"[green]✓ 加载完成: {analysis.project_name or '未命名项目'}[/green]")

        # Mode 2: Interactive mode
        elif interactive:
            output.print("[dim]📄 解析文档...[/dim]")
            doc = analyzer.parse_document(file_path)

            if doc.has_images():
                output.print(f"[green]✓ 文档解析完成 (包含 {len(doc.images)} 张图片)[/green]")
            else:
                output.print("[green]✓ 文档解析完成[/green]")

            # Output prompt to stdout
            prompt = analyzer.get_extraction_prompt(doc)
            output.print("\n[bold yellow]=" * 60 + "[/bold yellow]")
            output.print(
                "[bold yellow]请将以下内容交给 LLM 分析，然后输入 JSON 结果：[/bold yellow]"
            )
            output.print("[bold yellow]=" * 60 + "[/bold yellow]\n")

            # Print prompt to stdout (for piping to LLM)
            print(prompt)

            output.print("\n[bold yellow]=" * 60 + "[/bold yellow]")
            output.print("[bold yellow]请输入 LLM 返回的 JSON (以空行结束)：[/bold yellow]")
            output.print("[bold yellow]=" * 60 + "[/bold yellow]\n")

            # Read JSON from stdin
            json_lines = []
            try:
                for line in sys.stdin:
                    if line.strip() == "":
                        break
                    json_lines.append(line)
            except EOFError:
                pass

            json_text = "".join(json_lines)
            if not json_text.strip():
                output.print("[red]错误: 未收到 JSON 输入[/red]")
                return

            # Parse JSON
            try:
                # Try to extract JSON from markdown code block
                import re

                json_match = re.search(r"```json\s*(.*?)\s*```", json_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1)
                else:
                    json_match = re.search(r"\{.*\}", json_text, re.DOTALL)
                    if json_match:
                        json_text = json_match.group(0)

                extracted = json.loads(json_text)
                analysis = analyzer.create_analysis_from_json(extracted, doc)
                output.print(
                    f"[green]✓ JSON 解析成功: {analysis.project_name or '未命名项目'}[/green]"
                )
            except json.JSONDecodeError as e:
                output.print(f"[red]错误: JSON 解析失败 - {e}[/red]")
                return

        # Mode 3: API mode (default)
        else:
            output.print("[dim]📄 解析文档...[/dim]")
            analysis = analyzer.analyze(file_path)

            if analysis.has_images:
                output.print(f"[green]✓ 文档解析完成 (包含 {analysis.image_count} 张图片)[/green]")
            else:
                output.print("[green]✓ 文档解析完成[/green]")

            output.print("[dim]🤖 使用 LLM 提取结构化信息...[/dim]")
            if analysis.project_name:
                output.print(f"[green]✓ 识别项目: {analysis.project_name}[/green]")
                output.print(f"  类型: {analysis.dataset_type or 'unknown'}")
                output.print(f"  难度: {analysis.estimated_difficulty or 'unknown'}")
                output.print(f"  人工占比: {analysis.estimated_human_percentage:.0f}%")
            else:
                output.print("[yellow]⚠ LLM 提取信息有限，将使用默认值[/yellow]")

        # Step 3: LLM Enhancement (optional, enriches document quality)
        enhanced_context = None
        try:
            from datarecipe.generators.llm_enhancer import LLMEnhancer

            enhance_mode = "api" if not interactive else "interactive"
            enhancer = LLMEnhancer(mode=enhance_mode, provider=provider)
            enhanced_context = enhancer.enhance(
                dataset_id=analysis.project_name or "spec_analysis",
                dataset_type=analysis.dataset_type or "unknown",
                domain=analysis.estimated_domain or "通用",
                difficulty=analysis.estimated_difficulty or "medium",
                human_percentage=analysis.estimated_human_percentage,
                total_cost=0,
            )
            if enhanced_context and enhanced_context.generated:
                output.print("[green]✓ LLM 增强完成[/green]")
        except Exception:
            pass

        # Step 4: Generate outputs
        output.print("[dim]📝 生成项目文档...[/dim]")
        generator = SpecOutputGenerator(output_dir=output_dir)
        result = generator.generate(
            analysis=analysis,
            target_size=size,
            region=region,
            enhanced_context=enhanced_context,
        )

        if not result.success:
            output.print(f"[red]错误: {result.error}[/red]")
            return

        output.print("[green]✓ 生成完成[/green]")

        # Display summary
        output.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
        output.print("[bold cyan]  分析完成[/bold cyan]")
        output.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")

        output.print("[bold]生成的文件:[/bold]")
        for fname in result.files_generated:
            fpath = os.path.join(result.output_dir, fname)
            if os.path.exists(fpath):
                fsize = os.path.getsize(fpath)
                if fsize > 1024:
                    size_str = f"{fsize / 1024:.1f}KB"
                else:
                    size_str = f"{fsize}B"
                icon = "📊" if fname.endswith(".json") else "📄" if fname.endswith(".md") else "📑"
                output.print(f"  {icon} {fname} ({size_str})")

        output.print(f"\n[bold]输出目录:[/bold] [cyan]{result.output_dir}[/cyan]")

        # Key files
        output.print("\n[bold]核心产出:[/bold]")
        output.print(
            f"  📄 执行摘要: [cyan]{result.output_dir}/01_决策参考/EXECUTIVE_SUMMARY.md[/cyan]"
        )
        output.print(
            f"  📋 里程碑计划: [cyan]{result.output_dir}/02_项目管理/MILESTONE_PLAN.md[/cyan]"
        )
        output.print(
            f"  📝 标注规范: [cyan]{result.output_dir}/03_标注规范/ANNOTATION_SPEC.md[/cyan]"
        )

    except FileNotFoundError as e:
        output.print(f"[red]错误: 文件未找到 - {e}[/red]")
    except ValueError as e:
        output.print(f"[red]错误: {e}[/red]")
    except ImportError as e:
        output.print(f"[red]错误: 缺少依赖 - {e}[/red]")
        output.print("[dim]请安装所需依赖: pip install anthropic pymupdf python-docx[/dim]")
    except Exception as e:
        output.print(f"[red]错误: {e}[/red]")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    analyze_spec()
