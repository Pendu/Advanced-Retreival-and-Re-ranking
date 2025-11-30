# Documentation

This directory contains the Sphinx documentation for the Advanced Retrieval and Re-ranking project.

## Building Documentation Locally

### Prerequisites

Make sure you have the required packages installed:

```bash
conda activate myenv
pip install sphinx sphinx-wagtail-theme
```

### Build HTML Documentation

**On Windows (PowerShell):**

```powershell
cd docs
sphinx-build -b html source build/html
```

**On Linux/Mac:**

```bash
cd docs
make html
```

The built documentation will be in `docs/build/html/`. Open `index.html` in your browser to view it.

## Deploying to GitHub Pages

This repository includes a GitHub Actions workflow that automatically builds and deploys 
the documentation to GitHub Pages when you push to the main branch.

### Setup Steps

1. **Enable GitHub Pages** in your repository settings:
   - Go to Settings → Pages
   - Under "Source", select "GitHub Actions"

2. **Push your changes** to the main branch:
   ```bash
   git add .
   git commit -m "Add Sphinx documentation"
   git push origin main
   ```

3. **Wait for the workflow** to complete:
   - Go to the "Actions" tab in your repository
   - Watch the "Build and Deploy Documentation" workflow run
   - Once complete, your docs will be available at:
     `https://yourusername.github.io/Advanced-Retreival-and-Re-ranking/`

### Workflow Details

The `.github/workflows/docs.yml` file contains the automation that:
- Installs Python and dependencies
- Builds the Sphinx documentation
- Creates a `.nojekyll` file (required for GitHub Pages)
- Deploys to GitHub Pages

## Documentation Structure

```
docs/
├── source/                  # Source files for documentation
│   ├── conf.py             # Sphinx configuration
│   ├── index.rst           # Main documentation page
│   ├── overview.rst        # Overview and paradigm shift
│   ├── hard_negatives.rst  # Hard negative problem explanation
│   ├── papers.rst          # Research papers collection
│   ├── contributing.rst    # Contribution guidelines
│   ├── _static/            # Static files (CSS, images, etc.)
│   └── _templates/         # Custom templates
├── build/                   # Built documentation (generated)
│   └── html/               # HTML output
├── Makefile                # Build automation (Linux/Mac)
└── README.md               # This file
```

## Customization

### Theme Configuration

The documentation uses the Wagtail theme. You can customize it in `source/conf.py`:

```python
html_theme_options = {
    'project_name': 'Your Project Name',
    'github_url': 'https://github.com/yourusername/yourrepo',
    # Add more options as needed
}
```

### Adding New Pages

1. Create a new `.rst` file in `source/`
2. Add it to the `toctree` in `source/index.rst`
3. Rebuild the documentation

## Troubleshooting

### Theme Not Found

If you get "no theme named 'sphinx_wagtail_theme' found", make sure:
1. The theme is installed: `pip install sphinx-wagtail-theme`
2. You're using the correct conda environment: `conda activate myenv`
3. The import is in `conf.py`: `import sphinx_wagtail_theme`

### Build Errors

- Check that all `.rst` files have valid reStructuredText syntax
- Ensure all internal links reference existing pages
- Verify that all external links are properly formatted

### GitHub Pages Not Updating

1. Check the Actions tab for workflow errors
2. Ensure GitHub Pages is set to use "GitHub Actions" as the source
3. Verify that the workflow file is in `.github/workflows/`
4. Check that you have the necessary permissions in repository settings

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Wagtail Theme Documentation](https://github.com/wagtail/sphinx-wagtail-theme)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)

