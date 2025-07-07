---
trigger: always_on
---

# Development Guidelines

## Documentation First Approach

### Before Starting Any Task
1. **Always check the documentation first** in the `docs/` folder for:
   - Existing implementations that can be reused
   - Established patterns and conventions
   - System architecture and component interactions

2. **Update documentation** when making significant changes to:
   - Code structure
   - APIs
   - Data formats
   - Configuration options

### Documentation Structure
- `docs/index.md` - Project overview and navigation
- `docs/features/` - Feature documentation
- `docs/data_pipeline/` - Data processing details
- `docs/models/` - Model architecture and training
- `docs/api/` - API references and usage

### When to Update Documentation
- Adding new features
- Changing existing behavior
- Fixing bugs that reveal documentation gaps
- Receiving feedback about unclear documentation

## Code Review Checklist
- [ ] Documentation is up to date
- [ ] Follows project coding standards
- [ ] Includes appropriate tests
- [ ] No sensitive data in commits
- [ ] Proper error handling

## Best Practices
1. **For New Features**:
   - Create a new markdown file in the relevant docs directory
   - Include examples and usage patterns
   - Document any configuration options

2. **For Bug Fixes**:
   - Document the issue and solution
   - Add test cases if applicable
   - Update any affected documentation

3. **For Dependencies**:
   - Document any new dependencies
   - Update requirements files
   - Note any compatibility considerations

## Maintaining Documentation Quality
- Use clear, concise language
- Include code examples where helpful
- Keep diagrams and architecture documents current
- Review documentation during code reviews

Remember: Good documentation is as important as good code. If it's not documented, it doesn't exist!
