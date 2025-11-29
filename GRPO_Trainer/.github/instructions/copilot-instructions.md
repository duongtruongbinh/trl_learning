# Copilot Instructions

## Code Style Rules

- Do NOT use emoji or icons in code, comments, or documentation
- Do NOT use try-except blocks unless explicitly requested
- Do NOT add description comments or docstrings
- Keep code minimal and clean
- Write code directly without explanations
- No decorative elements in output

## Examples

### Do NOT write:
```python
def process_data(data):
    """
    Process the input data and return results.
    
    Args:
        data: Input data to process
    Returns:
        Processed data
    """
    try:
        result = data * 2
        return result
    except Exception as e:
        print(f"Error: {e}")
```

### Write instead:
```python
def process_data(data):
    result = data * 2
    return result
```

## Summary

1. No icons/emoji
2. No try-except
3. No descriptions/docstrings
4. Minimal, clean code only