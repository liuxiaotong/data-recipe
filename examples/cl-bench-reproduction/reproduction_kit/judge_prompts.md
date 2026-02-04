# 评估 Judge Prompt 模板

## binary_rubric_check

```
You are an expert evaluator. Your task is to determine whether the given response satisfies a specific requirement (rubric).

## Response to Evaluate
{response}

## Rubric (Requirement to Check)
{rubric}

## Instructions
1. Carefully read the response
2. Check if the response satisfies the rubric requirement
3. The rubric may require specific information, format, or behavior
4. Be strict: partial satisfaction is NOT sufficient

Answer with ONLY "yes" or "no".
```

## multi_rubric_batch

```
You are an expert evaluator. Evaluate whether the response satisfies EACH of the following rubrics.

## Response
{response}

## Rubrics to Check
{rubrics_numbered}

## Instructions
For each rubric, determine if the response satisfies it.
- Be strict: the response must fully meet the requirement
- Partial satisfaction counts as "no"

Output format (JSON):
{{
  "1": "yes" or "no",
  "2": "yes" or "no",
  ...
}}
```

## detailed_evaluation

```
You are an expert evaluator assessing an AI response against specific criteria.

## Context
The AI was given a task that requires learning from the provided context and applying that knowledge correctly.

## Original Task Context (Summary)
{context_summary}

## AI Response
{response}

## Evaluation Rubrics
{rubrics_list}

## Instructions
1. For each rubric, determine if the response satisfies it
2. Provide a brief explanation for each judgment
3. Calculate the overall score

Output format:
{{
  "rubric_results": [
    {{"rubric": "...", "satisfied": true/false, "reason": "..."}},
    ...
  ],
  "total_satisfied": N,
  "total_rubrics": M,
  "score": N/M,
  "passed": true/false (true only if ALL rubrics satisfied)
}}
```

