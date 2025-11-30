# Quick Reference Card

## 🏗️ Build Documentation Locally

```bash
conda activate myenv
cd docs
sphinx-build -b html source build/html
```

View at: `docs/build/html/index.html`

## 🚀 Deploy to GitHub Pages

```bash
# First time setup
git init
git add .
git commit -m "Initial commit with documentation"
git remote add origin https://github.com/YOURUSERNAME/Advanced-Retreival-and-Re-ranking.git
git branch -M main
git push -u origin main
```

Then: Settings → Pages → Source: "GitHub Actions"

## 📝 Update Documentation

```bash
# Edit .rst files in docs/source/
# Then commit and push:
git add docs/source/
git commit -m "Update documentation"
git push origin main
```

Docs auto-rebuild and deploy in ~2-3 minutes!

## 📂 Key Files to Edit

| File | Purpose |
|------|---------|
| `docs/source/conf.py` | Sphinx config, theme settings |
| `docs/source/index.rst` | Main landing page |
| `docs/source/papers.rst` | Research papers table |
| `docs/source/overview.rst` | Overview content |
| `docs/source/hard_negatives.rst` | Problem explanation |
| `docs/source/contributing.rst` | Contribution guidelines |

## 🔧 Common Tasks

### Add a New Paper

1. Edit `docs/source/papers.rst`
2. Add row to table with paper info
3. Commit and push

### Add a New Section

1. Create `docs/source/newsection.rst`
2. Add to toctree in `docs/source/index.rst`:
   ```rst
   .. toctree::
      :maxdepth: 2
      
      overview
      hard_negatives
      papers
      newsection    ← Add here
      contributing
   ```
3. Commit and push

### Change Author Name

Edit `docs/source/conf.py`:
```python
author = 'Your Name'  # Line 11
```

### Update GitHub URL

Edit `docs/source/conf.py`:
```python
'github_url': 'https://github.com/YOURUSERNAME/repo',  # Line 37
```

## 🌐 Access Documentation

After deployment:
```
https://YOURUSERNAME.github.io/Advanced-Retreival-and-Re-ranking/
```

## 🐛 Troubleshooting

### Build fails locally
- Check for syntax errors in .rst files
- Ensure theme is installed: `pip install sphinx-wagtail-theme`
- Verify you're in correct environment: `conda activate myenv`

### GitHub Actions fails
- Check Actions tab for error logs
- Verify `.github/workflows/docs.yml` exists
- Ensure dependencies in workflow match local

### Pages not updating
- Hard refresh: Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac)
- Check Actions tab for successful build
- Verify Pages setting: Settings → Pages → Source: "GitHub Actions"

## 📊 Monitor Deployment

Watch deployment:
1. Go to Actions tab
2. Click latest workflow run
3. View logs

Add badge to README:
```markdown
![Docs](https://github.com/YOURUSERNAME/Advanced-Retreival-and-Re-ranking/actions/workflows/docs.yml/badge.svg)
```

## 📚 Documentation URLs

- **Sphinx**: https://www.sphinx-doc.org/
- **Wagtail Theme**: https://github.com/wagtail/sphinx-wagtail-theme
- **reStructuredText**: https://docutils.sourceforge.io/rst.html
- **GitHub Pages**: https://docs.github.com/en/pages

## ⚡ Quick Commands

```bash
# Build docs
cd docs && sphinx-build -b html source build/html

# Clean build
rm -rf docs/build/html && cd docs && sphinx-build -b html source build/html

# Check links
cd docs && sphinx-build -b linkcheck source build/linkcheck

# View in browser (Windows)
explorer docs/build/html/index.html

# View in browser (Mac)
open docs/build/html/index.html

# View in browser (Linux)
xdg-open docs/build/html/index.html
```

## 🎯 File You Created

- [x] Sphinx documentation structure
- [x] Wagtail theme configuration
- [x] Content from README (4 pages)
- [x] GitHub Actions workflow
- [x] .nojekyll file
- [x] .gitignore
- [x] Setup guides

**Total**: ~10 files created, documentation ready to deploy!

