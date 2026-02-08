"""Unit tests for workflow.py - production workflow generation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from datarecipe.cost_calculator import CostBreakdown, CostEstimate
from datarecipe.schema import GenerationType, Recipe, SourceType
from datarecipe.workflow import (
    Milestone,
    ProductionWorkflow,
    ResourceChecklist,
    WorkflowGenerator,
    WorkflowStep,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_recipe(**overrides) -> Recipe:
    """Create a Recipe with sensible defaults, overridden by kwargs."""
    defaults = {
        "name": "test-dataset",
        "source_type": SourceType.HUGGINGFACE,
        "source_id": "org/test-dataset",
        "num_examples": 1000,
        "generation_type": GenerationType.SYNTHETIC,
        "synthetic_ratio": 1.0,
        "human_ratio": 0.0,
        "teacher_models": ["gpt-4o"],
    }
    defaults.update(overrides)
    return Recipe(**defaults)


def _make_cost_breakdown(**overrides) -> CostBreakdown:
    """Create a CostBreakdown with zero-cost defaults."""
    api = overrides.pop("api_cost", CostEstimate(0, 0, 0))
    human = overrides.pop("human_annotation_cost", CostEstimate(0, 0, 0))
    compute = overrides.pop("compute_cost", CostEstimate(0, 0, 0))
    total = overrides.pop("total", CostEstimate(0, 0, 0))
    return CostBreakdown(
        api_cost=api,
        human_annotation_cost=human,
        compute_cost=compute,
        total=total,
        **overrides,
    )


# ===========================================================================
# Dataclass tests
# ===========================================================================

class TestWorkflowStep(unittest.TestCase):
    """Tests for the WorkflowStep dataclass."""

    def test_defaults(self):
        step = WorkflowStep(name="s1", description="d1", script_content="pass")
        self.assertEqual(step.name, "s1")
        self.assertEqual(step.dependencies, [])
        self.assertEqual(step.env_vars, [])
        self.assertEqual(step.estimated_cost, 0.0)
        self.assertEqual(step.inputs, [])
        self.assertEqual(step.outputs, [])

    def test_custom_values(self):
        step = WorkflowStep(
            name="Generate",
            description="Generate data",
            script_content="print('hi')",
            dependencies=["openai"],
            env_vars=["OPENAI_API_KEY"],
            estimated_cost=100.0,
            inputs=["seed.json"],
            outputs=["out.jsonl"],
        )
        self.assertEqual(step.dependencies, ["openai"])
        self.assertEqual(step.env_vars, ["OPENAI_API_KEY"])
        self.assertAlmostEqual(step.estimated_cost, 100.0)


class TestMilestone(unittest.TestCase):
    def test_defaults(self):
        m = Milestone(name="M1", description="First", deliverables=["d1"])
        self.assertEqual(m.dependencies, [])

    def test_with_dependencies(self):
        m = Milestone(name="M2", description="Second", deliverables=["d2"], dependencies=["M1"])
        self.assertEqual(m.dependencies, ["M1"])


class TestResourceChecklist(unittest.TestCase):
    def test_defaults(self):
        rc = ResourceChecklist()
        self.assertEqual(rc.api_keys, [])
        self.assertEqual(rc.dependencies, [])
        self.assertEqual(rc.compute_requirements, {})

    def test_custom(self):
        rc = ResourceChecklist(
            api_keys=[("OpenAI", "https://openai.com", "OPENAI_API_KEY")],
            dependencies=["openai"],
            compute_requirements={"GPU": "A100"},
        )
        self.assertEqual(len(rc.api_keys), 1)
        self.assertEqual(rc.dependencies, ["openai"])


# ===========================================================================
# ProductionWorkflow tests
# ===========================================================================

class TestProductionWorkflow(unittest.TestCase):
    """Tests for the ProductionWorkflow dataclass and its methods."""

    def _make_workflow(self, num_steps=2, **kwargs):
        steps = []
        for i in range(num_steps):
            steps.append(
                WorkflowStep(
                    name=f"Step {i+1}",
                    description=f"Description {i+1}",
                    script_content=f"# step {i+1}\nprint('step {i+1}')",
                    dependencies=["dep-a"] if i == 0 else [],
                    env_vars=["MY_KEY"] if i == 0 else [],
                    estimated_cost=10.0 * (i + 1),
                    inputs=[f"in{i}.txt"],
                    outputs=[f"out{i}.txt"],
                )
            )
        defaults = {
            "name": "Test Workflow",
            "description": "A test workflow",
            "steps": steps,
            "milestones": [
                Milestone(name="M1", description="First milestone", deliverables=["d1", "d2"]),
            ],
            "resource_checklist": ResourceChecklist(
                api_keys=[("SomeAPI", "https://example.com", "SOME_KEY")],
                dependencies=["dep-a"],
                compute_requirements={"CPU": "4 cores"},
            ),
            "timeline": [("Phase 1", "Setup"), ("Phase 2", "Execute")],
            "estimated_total_cost": 100.0,
            "target_size": 5000,
        }
        defaults.update(kwargs)
        return ProductionWorkflow(**defaults)

    # -- generate_scripts ---------------------------------------------------

    def test_generate_scripts_creates_files(self):
        wf = self._make_workflow(num_steps=3)
        with tempfile.TemporaryDirectory() as tmp:
            files = wf.generate_scripts(tmp)
            self.assertEqual(len(files), 3)
            for f in files:
                self.assertTrue(Path(f).exists())
                # file should be inside a scripts/ subdirectory
                self.assertIn("scripts", f)

    def test_generate_scripts_content(self):
        wf = self._make_workflow(num_steps=1)
        with tempfile.TemporaryDirectory() as tmp:
            files = wf.generate_scripts(tmp)
            content = Path(files[0]).read_text()
            self.assertIn("#!/usr/bin/env python3", content)
            self.assertIn("Step 1: Step 1", content)
            self.assertIn("print('step 1')", content)

    def test_generate_scripts_naming(self):
        wf = self._make_workflow(num_steps=2)
        with tempfile.TemporaryDirectory() as tmp:
            files = wf.generate_scripts(tmp)
            names = [Path(f).name for f in files]
            self.assertEqual(names[0], "01_step_1.py")
            self.assertEqual(names[1], "02_step_2.py")

    def test_generate_scripts_no_steps(self):
        wf = self._make_workflow(num_steps=0)
        with tempfile.TemporaryDirectory() as tmp:
            files = wf.generate_scripts(tmp)
            self.assertEqual(files, [])

    # -- export_project -----------------------------------------------------

    def test_export_project_creates_structure(self):
        wf = self._make_workflow(num_steps=1)
        with tempfile.TemporaryDirectory() as tmp:
            files = wf.export_project(tmp)
            # Should create: script(s), README, requirements.txt, config.yaml,
            # checklist.md, timeline.md, .env.example
            basenames = [Path(f).name for f in files]
            self.assertIn("README.md", basenames)
            self.assertIn("requirements.txt", basenames)
            self.assertIn("config.yaml", basenames)
            self.assertIn("checklist.md", basenames)
            self.assertIn("timeline.md", basenames)
            # .env.example should exist because step has env_vars
            self.assertIn(".env.example", basenames)
            # data/ directory created (but not listed in files)
            self.assertTrue((Path(tmp) / "data").is_dir())

    def test_export_project_requirements_content(self):
        wf = self._make_workflow(num_steps=1)
        with tempfile.TemporaryDirectory() as tmp:
            wf.export_project(tmp)
            req = (Path(tmp) / "requirements.txt").read_text()
            # Should always include tqdm and datasets
            self.assertIn("tqdm", req)
            self.assertIn("datasets", req)
            # Should include step dependency
            self.assertIn("dep-a", req)

    def test_export_project_no_env_vars(self):
        """When no env_vars, .env.example should not be created."""
        step = WorkflowStep(name="plain", description="d", script_content="pass")
        wf = ProductionWorkflow(
            name="w", description="d", steps=[step],
            resource_checklist=ResourceChecklist(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            files = wf.export_project(tmp)
            basenames = [Path(f).name for f in files]
            self.assertNotIn(".env.example", basenames)

    def test_export_project_env_from_resource_checklist(self):
        """env_vars from resource_checklist api_keys should appear in .env.example."""
        step = WorkflowStep(name="s", description="d", script_content="pass")
        rc = ResourceChecklist(
            api_keys=[("MyAPI", "https://example.com", "MY_API_KEY")]
        )
        wf = ProductionWorkflow(
            name="w", description="d", steps=[step], resource_checklist=rc,
        )
        with tempfile.TemporaryDirectory() as tmp:
            files = wf.export_project(tmp)
            basenames = [Path(f).name for f in files]
            self.assertIn(".env.example", basenames)
            env_path = Path(tmp) / ".env.example"
            self.assertIn("MY_API_KEY", env_path.read_text())

    # -- _generate_readme ---------------------------------------------------

    def test_generate_readme(self):
        wf = self._make_workflow()
        readme = wf._generate_readme()
        self.assertIn("# Test Workflow", readme)
        self.assertIn("A test workflow", readme)
        self.assertIn("5,000 examples", readme)
        self.assertIn("$100", readme)
        self.assertIn("Step 1: Step 1", readme)
        self.assertIn("Step 2: Step 2", readme)
        self.assertIn("Generated by DataRecipe", readme)

    def test_readme_step_with_no_cost(self):
        step = WorkflowStep(
            name="Free", description="No cost", script_content="pass",
            estimated_cost=0.0,
        )
        wf = ProductionWorkflow(
            name="W", description="D", steps=[step], target_size=100,
        )
        readme = wf._generate_readme()
        # The Overview section always shows total estimated cost, but
        # the per-step "**Estimated Cost**:" line should NOT appear for a free step.
        # Count occurrences: only the overview one should exist.
        self.assertEqual(readme.count("**Estimated Cost**"), 1)
        # And that one occurrence is in the overview (formatted differently)
        self.assertIn("**Estimated Cost**: $0", readme)

    def test_readme_step_with_inputs_outputs(self):
        step = WorkflowStep(
            name="IO", description="Has IO", script_content="pass",
            inputs=["a.txt"], outputs=["b.txt"],
        )
        wf = ProductionWorkflow(name="W", description="D", steps=[step])
        readme = wf._generate_readme()
        self.assertIn("**Inputs**: a.txt", readme)
        self.assertIn("**Outputs**: b.txt", readme)

    # -- _generate_config ---------------------------------------------------

    def test_generate_config(self):
        wf = self._make_workflow()
        config_str = wf._generate_config()
        self.assertIn("Test Workflow", config_str)
        self.assertIn("target_size", config_str)
        self.assertIn("batch_size", config_str)

    # -- _generate_checklist ------------------------------------------------

    def test_generate_checklist_with_api_keys(self):
        wf = self._make_workflow()
        checklist = wf._generate_checklist()
        self.assertIn("SomeAPI", checklist)
        self.assertIn("SOME_KEY", checklist)
        self.assertIn("dep-a", checklist)
        self.assertIn("4 cores", checklist)

    def test_generate_checklist_no_api_keys(self):
        wf = ProductionWorkflow(
            name="W", description="D",
            resource_checklist=ResourceChecklist(),
            target_size=100,
        )
        checklist = wf._generate_checklist()
        self.assertIn("No API keys required.", checklist)

    def test_generate_checklist_no_compute(self):
        wf = ProductionWorkflow(
            name="W", description="D",
            resource_checklist=ResourceChecklist(),
            target_size=100,
        )
        checklist = wf._generate_checklist()
        self.assertIn("Standard CPU is sufficient", checklist)

    def test_generate_checklist_disk_space(self):
        wf = ProductionWorkflow(
            name="W", description="D",
            resource_checklist=ResourceChecklist(),
            target_size=50000,
        )
        checklist = wf._generate_checklist()
        self.assertIn("5GB", checklist)

    # -- _generate_timeline -------------------------------------------------

    def test_generate_timeline_with_milestones_and_phases(self):
        wf = self._make_workflow()
        timeline = wf._generate_timeline()
        self.assertIn("# Project Timeline", timeline)
        self.assertIn("M1: M1", timeline)
        self.assertIn("d1", timeline)
        self.assertIn("Phase 1", timeline)
        self.assertIn("Setup", timeline)
        # Step-by-step progress
        self.assertIn("Step 1: Step 1", timeline)

    def test_generate_timeline_no_milestones(self):
        wf = ProductionWorkflow(
            name="W", description="D",
            timeline=[("P1", "Do stuff")],
        )
        timeline = wf._generate_timeline()
        self.assertNotIn("## Milestones", timeline)
        self.assertIn("## Phases", timeline)

    def test_generate_timeline_no_phases(self):
        wf = ProductionWorkflow(
            name="W", description="D",
            milestones=[Milestone(name="M1", description="D", deliverables=["x"])],
        )
        timeline = wf._generate_timeline()
        self.assertIn("## Milestones", timeline)
        self.assertNotIn("## Phases", timeline)


# ===========================================================================
# WorkflowGenerator tests
# ===========================================================================

class TestWorkflowGenerator(unittest.TestCase):
    """Tests for WorkflowGenerator.generate() and its sub-methods."""

    def setUp(self):
        self.generator = WorkflowGenerator()

    # -- routing based on synthetic_ratio -----------------------------------

    @patch.object(WorkflowGenerator, "_generate_synthetic_workflow")
    def test_generate_routes_to_synthetic(self, mock_syn):
        mock_syn.return_value = ProductionWorkflow(name="syn", description="d")
        recipe = _make_recipe(synthetic_ratio=0.95)
        result = self.generator.generate(recipe)
        mock_syn.assert_called_once()
        self.assertEqual(result.name, "syn")

    @patch.object(WorkflowGenerator, "_generate_human_workflow")
    def test_generate_routes_to_human(self, mock_human):
        mock_human.return_value = ProductionWorkflow(name="human", description="d")
        recipe = _make_recipe(synthetic_ratio=0.05, human_ratio=0.95)
        result = self.generator.generate(recipe)
        mock_human.assert_called_once()
        self.assertEqual(result.name, "human")

    @patch.object(WorkflowGenerator, "_generate_hybrid_workflow")
    def test_generate_routes_to_hybrid(self, mock_hybrid):
        mock_hybrid.return_value = ProductionWorkflow(name="hybrid", description="d")
        recipe = _make_recipe(synthetic_ratio=0.5, human_ratio=0.5)
        result = self.generator.generate(recipe)
        mock_hybrid.assert_called_once()
        self.assertEqual(result.name, "hybrid")

    # -- target size fallback -----------------------------------------------

    @patch.object(WorkflowGenerator, "_generate_synthetic_workflow")
    def test_target_size_from_argument(self, mock_syn):
        mock_syn.return_value = ProductionWorkflow(name="x", description="d")
        recipe = _make_recipe(synthetic_ratio=1.0, num_examples=500)
        self.generator.generate(recipe, target_size=2000)
        args = mock_syn.call_args
        self.assertEqual(args[0][1], 2000)  # second positional arg = target_size

    @patch.object(WorkflowGenerator, "_generate_synthetic_workflow")
    def test_target_size_from_recipe(self, mock_syn):
        mock_syn.return_value = ProductionWorkflow(name="x", description="d")
        recipe = _make_recipe(synthetic_ratio=1.0, num_examples=777)
        self.generator.generate(recipe)
        args = mock_syn.call_args
        self.assertEqual(args[0][1], 777)

    @patch.object(WorkflowGenerator, "_generate_synthetic_workflow")
    def test_target_size_default(self, mock_syn):
        mock_syn.return_value = ProductionWorkflow(name="x", description="d")
        recipe = _make_recipe(synthetic_ratio=1.0, num_examples=None)
        self.generator.generate(recipe)
        args = mock_syn.call_args
        self.assertEqual(args[0][1], 10000)

    # -- _generate_synthetic_workflow ---------------------------------------

    def test_synthetic_workflow_structure(self):
        recipe = _make_recipe(synthetic_ratio=1.0, teacher_models=["gpt-4o"])
        wf = self.generator._generate_synthetic_workflow(recipe, 1000, "huggingface")
        self.assertEqual(len(wf.steps), 5)
        step_names = [s.name for s in wf.steps]
        self.assertIn("Seed Data", step_names)
        self.assertIn("LLM Generation", step_names)
        self.assertIn("Quality Filtering", step_names)
        self.assertIn("Deduplication", step_names)
        self.assertIn("Validation", step_names)

    def test_synthetic_workflow_openai_api_key(self):
        recipe = _make_recipe(synthetic_ratio=1.0, teacher_models=["gpt-4o"])
        wf = self.generator._generate_synthetic_workflow(recipe, 1000, "huggingface")
        api_key_names = [name for name, _, _ in wf.resource_checklist.api_keys]
        self.assertIn("OpenAI API", api_key_names)

    def test_synthetic_workflow_anthropic_api_key(self):
        recipe = _make_recipe(synthetic_ratio=1.0, teacher_models=["claude-3-sonnet"])
        wf = self.generator._generate_synthetic_workflow(recipe, 1000, "huggingface")
        api_key_names = [name for name, _, _ in wf.resource_checklist.api_keys]
        self.assertIn("Anthropic API", api_key_names)

    def test_synthetic_workflow_milestones(self):
        recipe = _make_recipe(synthetic_ratio=1.0)
        wf = self.generator._generate_synthetic_workflow(recipe, 1000, "huggingface")
        self.assertEqual(len(wf.milestones), 4)
        milestone_names = [m.name for m in wf.milestones]
        self.assertIn("Setup Complete", milestone_names)
        self.assertIn("Generation Complete", milestone_names)
        self.assertIn("Quality Assured", milestone_names)
        self.assertIn("Dataset Published", milestone_names)

    def test_synthetic_workflow_timeline(self):
        recipe = _make_recipe(synthetic_ratio=1.0)
        wf = self.generator._generate_synthetic_workflow(recipe, 1000, "huggingface")
        self.assertEqual(len(wf.timeline), 4)

    def test_synthetic_workflow_name_and_description(self):
        recipe = _make_recipe(name="my-ds", synthetic_ratio=1.0)
        wf = self.generator._generate_synthetic_workflow(recipe, 5000, "huggingface")
        self.assertIn("my-ds", wf.name)
        self.assertIn("5,000", wf.description)
        self.assertEqual(wf.target_size, 5000)

    def test_synthetic_workflow_cost_populated(self):
        recipe = _make_recipe(synthetic_ratio=1.0)
        wf = self.generator._generate_synthetic_workflow(recipe, 1000, "huggingface")
        self.assertIsInstance(wf.estimated_total_cost, float)

    def test_synthetic_workflow_default_model(self):
        """When teacher_models is empty, default to gpt-4o."""
        recipe = _make_recipe(synthetic_ratio=1.0, teacher_models=[])
        wf = self.generator._generate_synthetic_workflow(recipe, 1000, "huggingface")
        # LLM Generation step should reference gpt-4o
        gen_step = [s for s in wf.steps if s.name == "LLM Generation"][0]
        self.assertIn("gpt-4o", gen_step.description)

    # -- _generate_human_workflow -------------------------------------------

    def test_human_workflow_structure(self):
        recipe = _make_recipe(synthetic_ratio=0.0, human_ratio=1.0)
        wf = self.generator._generate_human_workflow(recipe, 2000, "huggingface")
        self.assertEqual(len(wf.steps), 5)
        step_names = [s.name for s in wf.steps]
        self.assertIn("Data Collection", step_names)
        self.assertIn("Annotation Guidelines", step_names)
        self.assertIn("Annotation Platform Setup", step_names)
        self.assertIn("Quality Control", step_names)
        self.assertIn("Dataset Export", step_names)

    def test_human_workflow_milestones(self):
        recipe = _make_recipe(synthetic_ratio=0.0, human_ratio=1.0)
        wf = self.generator._generate_human_workflow(recipe, 2000, "huggingface")
        self.assertEqual(len(wf.milestones), 4)

    def test_human_workflow_resource_checklist(self):
        recipe = _make_recipe(synthetic_ratio=0.0, human_ratio=1.0)
        wf = self.generator._generate_human_workflow(recipe, 2000, "huggingface")
        dep_names = wf.resource_checklist.dependencies
        self.assertIn("label-studio", dep_names)
        self.assertIn("pandas", dep_names)

    def test_human_workflow_timeline(self):
        recipe = _make_recipe(synthetic_ratio=0.0, human_ratio=1.0)
        wf = self.generator._generate_human_workflow(recipe, 2000, "huggingface")
        self.assertEqual(len(wf.timeline), 4)
        phases = [p for p, _ in wf.timeline]
        self.assertIn("Phase 1: Collection", phases)

    def test_human_workflow_name(self):
        recipe = _make_recipe(name="human-ds", synthetic_ratio=0.0, human_ratio=1.0)
        wf = self.generator._generate_human_workflow(recipe, 500, "huggingface")
        self.assertIn("human-ds", wf.name)

    # -- _generate_hybrid_workflow ------------------------------------------

    def test_hybrid_workflow_has_verification_step(self):
        recipe = _make_recipe(synthetic_ratio=0.5, human_ratio=0.5)
        wf = self.generator._generate_hybrid_workflow(recipe, 1000, "huggingface")
        step_names = [s.name for s in wf.steps]
        self.assertIn("Human Verification", step_names)

    def test_hybrid_workflow_name(self):
        recipe = _make_recipe(name="hybrid-ds", synthetic_ratio=0.5, human_ratio=0.5)
        wf = self.generator._generate_hybrid_workflow(recipe, 1000, "huggingface")
        self.assertIn("Hybrid", wf.name)

    def test_hybrid_workflow_description(self):
        recipe = _make_recipe(synthetic_ratio=0.5, human_ratio=0.5)
        wf = self.generator._generate_hybrid_workflow(recipe, 1000, "huggingface")
        self.assertIn("Hybrid workflow", wf.description)
        self.assertIn("1,000", wf.description)

    def test_hybrid_verification_step_cost(self):
        recipe = _make_recipe(synthetic_ratio=0.5, human_ratio=0.3)
        wf = self.generator._generate_hybrid_workflow(recipe, 1000, "huggingface")
        verification = [s for s in wf.steps if s.name == "Human Verification"][0]
        # cost = quality_check * target_size * human_ratio
        expected_cost = self.generator.cost_calculator.ANNOTATION_COSTS["quality_check"] * 1000 * 0.3
        self.assertAlmostEqual(verification.estimated_cost, expected_cost)

    def test_hybrid_inserts_verification_before_last(self):
        """Human Verification should be inserted just before the last step (Validation)."""
        recipe = _make_recipe(synthetic_ratio=0.5, human_ratio=0.5)
        wf = self.generator._generate_hybrid_workflow(recipe, 1000, "huggingface")
        # Verification should be second-to-last
        self.assertEqual(wf.steps[-2].name, "Human Verification")
        self.assertEqual(wf.steps[-1].name, "Validation")

    # -- edge cases ---------------------------------------------------------

    def test_synthetic_ratio_none(self):
        """synthetic_ratio=None should be treated as 0.0, routing to human."""
        recipe = _make_recipe(synthetic_ratio=None)
        # synthetic_ratio or 0.0 => 0.0, which is <= 0.1, so human workflow
        with patch.object(
            self.generator, "_generate_human_workflow",
            return_value=ProductionWorkflow(name="h", description="d"),
        ) as mock_h:
            self.generator.generate(recipe)
            mock_h.assert_called_once()

    def test_boundary_synthetic_ratio_090(self):
        """synthetic_ratio=0.9 is >= 0.9 so should use synthetic workflow."""
        recipe = _make_recipe(synthetic_ratio=0.9)
        with patch.object(
            self.generator, "_generate_synthetic_workflow",
            return_value=ProductionWorkflow(name="s", description="d"),
        ) as mock_s:
            self.generator.generate(recipe)
            mock_s.assert_called_once()

    def test_boundary_synthetic_ratio_010(self):
        """synthetic_ratio=0.1 is <= 0.1 so should use human workflow."""
        recipe = _make_recipe(synthetic_ratio=0.1)
        with patch.object(
            self.generator, "_generate_human_workflow",
            return_value=ProductionWorkflow(name="h", description="d"),
        ) as mock_h:
            self.generator.generate(recipe)
            mock_h.assert_called_once()


# ===========================================================================
# Integration tests (no mocking of CostCalculator)
# ===========================================================================

class TestWorkflowGeneratorIntegration(unittest.TestCase):
    """Integration tests that exercise the full generation pipeline."""

    def setUp(self):
        self.generator = WorkflowGenerator()

    def test_full_synthetic_workflow_export(self):
        recipe = _make_recipe(synthetic_ratio=1.0, teacher_models=["gpt-4o"])
        wf = self.generator.generate(recipe, target_size=500)
        with tempfile.TemporaryDirectory() as tmp:
            files = wf.export_project(tmp)
            self.assertTrue(len(files) > 5)
            # Verify all files actually exist
            for f in files:
                self.assertTrue(Path(f).exists(), f"File not found: {f}")

    def test_full_human_workflow_export(self):
        recipe = _make_recipe(synthetic_ratio=0.0, human_ratio=1.0)
        wf = self.generator.generate(recipe, target_size=500)
        with tempfile.TemporaryDirectory() as tmp:
            files = wf.export_project(tmp)
            self.assertTrue(len(files) > 5)

    def test_full_hybrid_workflow_export(self):
        recipe = _make_recipe(synthetic_ratio=0.5, human_ratio=0.5)
        wf = self.generator.generate(recipe, target_size=500)
        with tempfile.TemporaryDirectory() as tmp:
            files = wf.export_project(tmp)
            self.assertTrue(len(files) > 5)

    def test_output_format_argument_accepted(self):
        """output_format is passed through (currently unused, but should not break)."""
        recipe = _make_recipe(synthetic_ratio=1.0)
        wf = self.generator.generate(recipe, output_format="jsonl")
        self.assertIsInstance(wf, ProductionWorkflow)

    def test_generate_with_no_teacher_models(self):
        recipe = _make_recipe(synthetic_ratio=1.0, teacher_models=[])
        wf = self.generator.generate(recipe, target_size=200)
        self.assertIsInstance(wf, ProductionWorkflow)
        self.assertTrue(wf.estimated_total_cost >= 0)


if __name__ == "__main__":
    unittest.main()
