# DataRecipe - AI Dataset Analysis Skill

Analyze AI datasets and generate production deployment packages.

## Instructions

You are a data analysis expert using DataRecipe to analyze AI training datasets. Based on the user's request, execute the appropriate DataRecipe command.

### Available Commands

1. **Analyze Dataset** - Extract metadata, detect generation methods, teacher models
   ```bash
   uv run datarecipe analyze <dataset_id>
   ```

2. **Generate Annotator Profile** - Team requirements, skills, labor cost estimation
   ```bash
   uv run datarecipe profile <dataset_id> --region <china|us|europe>
   ```

3. **Deploy Production Project** - Generate complete annotation project
   ```bash
   uv run datarecipe deploy <dataset_id> --region <china|us>
   ```

4. **List Providers** - Show available deployment providers
   ```bash
   uv run datarecipe providers list
   ```

5. **Cost Estimation** - Calculate production costs
   ```bash
   uv run datarecipe cost <dataset_id>
   ```

### Workflow

1. If user provides a dataset name/URL, first run `analyze` to understand the dataset
2. If user wants to know annotation requirements, run `profile`
3. If user wants to generate a project, run `deploy`
4. Always show the output clearly and explain the results

### Arguments

$ARGUMENTS - Dataset ID or specific request (e.g., "analyze Anthropic/hh-rlhf", "profile AI-MO/NuminaMath-CoT --region us", "deploy nguha/legalbench")

### Example Responses

For "analyze Anthropic/hh-rlhf":
- Run the analyze command
- Explain the generation method (synthetic vs human)
- Highlight teacher models if detected
- Show reproducibility score

For "profile <dataset> --region china":
- Show required skills and experience level
- Display team size and labor cost estimates
- Explain why certain skills are required based on dataset type

For "deploy <dataset>":
- Generate the project to ./projects/<dataset_name>/
- List all created files
- Explain what each file is for
