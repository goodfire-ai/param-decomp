# Contributing to PD

Thank you for your interest in contributing to PD! Please follow these guidelines to ensure a smooth contribution process.

## Pull Request Guidelines

### 1. Use the Pull Request Template
Always use the pull request template located at `.github/pull_request_template.md`. This applies to both human contributors and AI assistants (if you're using an AI assistant like Claude, ensure it follows this template).

### 2. Link Related Issues
If your PR closes an existing issue, include "Closes #XX" in the Related Issue section of the PR template (where XX is the issue number).

### 3. Draft PRs for Work in Progress
If your PR is not ready for review, please make it a draft PR. This signals to maintainers that the work is still in progress.

### 4. Pre-Review Checklist
Before requesting a review on your PR, ensure:

a. **Review your diff**: Read every line of the diff your PR is making and ensure you're happy with all changes
b. **All checks pass**: You should see a green tick in your PR indicating all CI checks have passed
c. **Merge latest changes**: Merge the latest changes from the `dev` branch into your branch (unless there are no conflicts with the latest changes on dev)
d. **Optional AI review**: Consider asking Claude to review your PR with a comment like:
   > @claude can you review this? Just tell me potential bugs or issues and ways that it may be improved at a high or low level. I don't want to hear about ways that the PR is good.

### 5. Search Before Creating
Before making an issue or PR, search for existing issues or PRs in the repository. If you find related (but not duplicate) issues or PRs, reference them in your submission.

### 6. Review yourself before asking
Before asking for a PR review, read every line of the PR diff and ask yourself "should this line be in the PR?".

## Code Quality
- Run `make check` before committing to ensure your code passes all quality checks
- See [STYLE.md](./STYLE.md) for our coding style guide.
- Write clear commit messages that explain what changes were made and why
