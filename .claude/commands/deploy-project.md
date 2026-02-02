# Deploy Annotation Project

Generate a complete annotation project with guidelines, quality rules, and scripts.

## Instructions

Deploy an annotation project for the specified dataset.

```bash
uv run datarecipe deploy $ARGUMENTS
```

After deployment:
1. List all generated files and their purposes
2. Show the project timeline and milestones
3. Highlight key quality rules and acceptance criteria
4. Provide next steps for the user

If user doesn't specify output directory, files are saved to `./projects/<dataset_name>/`
