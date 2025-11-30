# 🚀 Deploy Your Documentation in 5 Minutes

## Quick Deploy Guide

### Step 1: Update Your GitHub Username (1 min)

**Edit `docs/source/conf.py`** (line 37):
```python
'github_url': 'https://github.com/YOURUSERNAME/Advanced-Retreival-and-Re-ranking',
```
Replace `YOURUSERNAME` with your actual GitHub username.

**Edit `docs/source/contributing.rst`** (near bottom):
```rst
https://github.com/YOURUSERNAME/Advanced-Retreival-and-Re-ranking
```
Replace `YOURUSERNAME` with your actual GitHub username.

### Step 2: Push to GitHub (2 min)

Open PowerShell in your project directory and run:

```powershell
# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Add Sphinx documentation with Wagtail theme"

# Add your GitHub repository
# (Replace YOURUSERNAME with your username)
git remote add origin https://github.com/YOURUSERNAME/Advanced-Retreival-and-Re-ranking.git

# Push to main branch
git branch -M main
git push -u origin main
```

**Don't have a repository yet?**
1. Go to https://github.com/new
2. Repository name: `Advanced-Retreival-and-Re-ranking`
3. Keep it Public
4. DON'T add README, .gitignore, or license (you already have them)
5. Click "Create repository"
6. Then run the commands above

### Step 3: Enable GitHub Pages (1 min)

1. Go to your repository on GitHub
2. Click **Settings** (top menu bar)
3. Click **Pages** in the left sidebar
4. Under "Build and deployment":
   - **Source**: Click dropdown and select **"GitHub Actions"**
5. That's it! No save button needed.

### Step 4: Wait for Deployment (2 min)

1. Click the **Actions** tab (top of repository)
2. You'll see "Build and Deploy Documentation" running (orange dot 🟠)
3. Wait for it to turn green (✅) - takes 2-3 minutes
4. Once green, click on it to see the deployment URL

### Step 5: View Your Documentation! (NOW!)

Your documentation is live at:
```
https://YOURUSERNAME.github.io/Advanced-Retreival-and-Re-ranking/
```

Replace `YOURUSERNAME` with your GitHub username and open that URL!

---

## ✅ Deployment Checklist

- [ ] Updated `docs/source/conf.py` with GitHub username
- [ ] Updated `docs/source/contributing.rst` with GitHub username
- [ ] Created GitHub repository
- [ ] Pushed code to GitHub (`git push origin main`)
- [ ] Enabled GitHub Pages (Settings → Pages → Source: GitHub Actions)
- [ ] Waited for green checkmark in Actions tab
- [ ] Visited the documentation URL
- [ ] Shared with the world! 🎉

---

## 🔧 Troubleshooting

### "Repository not found" when pushing
**Fix:** Make sure you created the repository on GitHub first at https://github.com/new

### "Permission denied" when pushing
**Fix:** You need to authenticate. Options:
1. Use HTTPS + Personal Access Token
2. Use SSH key
3. Use GitHub CLI: `gh auth login`

### Workflow doesn't run
**Fix:** 
- Check that `.github/workflows/docs.yml` exists
- Make sure you selected "GitHub Actions" (not "Deploy from a branch")
- Look at Actions tab for any error messages

### 404 Page Not Found
**Fix:**
- Wait 2-3 minutes for first deployment
- Check Actions tab - make sure deployment succeeded
- Verify URL format: `https://username.github.io/repo-name/` (trailing slash!)

### Build fails in Actions
**Fix:**
- Click on the failed workflow to see errors
- Common issue: Check that all .rst files have valid syntax
- Re-run the workflow (there's a "Re-run jobs" button)

---

## 🎨 Next Steps After Deployment

### Add Status Badge to README

Add to your `README.md`:

```markdown
![Documentation Status](https://github.com/YOURUSERNAME/Advanced-Retreival-and-Re-ranking/actions/workflows/docs.yml/badge.svg)

## Documentation

📖 Read the full documentation at: https://YOURUSERNAME.github.io/Advanced-Retreival-and-Re-ranking/
```

### Update Documentation

Every time you push to main, docs auto-rebuild:

```bash
# Edit any .rst files
# Then:
git add docs/source/
git commit -m "Update documentation"
git push origin main

# Wait 2-3 minutes, docs are updated!
```

### Customize the URL (Optional)

See **URL_OPTIONS.md** for:
- Renaming repository for shorter URL
- Setting up custom domain
- Using subdomain like `docs.yoursite.com`

---

## 📊 What Happens on Each Push?

```
You push to GitHub
       ↓
GitHub Actions triggers
       ↓
Ubuntu VM spins up
       ↓
Python 3.11 installed
       ↓
Sphinx + Wagtail theme installed
       ↓
Documentation built (sphinx-build)
       ↓
.nojekyll file created
       ↓
Deployed to GitHub Pages
       ↓
Live in 2-3 minutes! ✨
```

---

## 💡 Pro Tips

1. **Always check Actions tab** after pushing to ensure build succeeded
2. **Hard refresh** your browser (Ctrl+Shift+R) if you don't see changes
3. **Keep paper links updated** - automated deployment means updates are instant
4. **Monitor the Actions** - add email notifications in GitHub settings
5. **Use meaningful commit messages** - they show up in Actions history

---

## 🎯 Common Commands

```bash
# Build locally to test before pushing
cd docs
sphinx-build -b html source build/html

# Check if git is initialized
git status

# View your remote URL
git remote -v

# Check which branch you're on
git branch

# Push changes
git add .
git commit -m "Update docs"
git push origin main

# View build logs locally
cd docs && sphinx-build -b html source build/html
```

---

## 📱 Share Your Documentation

Once deployed, share your docs:

**On Twitter/X:**
```
🎉 Just published comprehensive documentation on Dense Retrieval and 
Re-ranking techniques!

📚 16 research papers with code implementations
🔬 Deep dive into hard negative sampling
🚀 Auto-deployed with GitHub Pages

Check it out: https://username.github.io/repo-name/

#MachineLearning #NLP #InformationRetrieval
```

**On LinkedIn:**
```
I've compiled and documented 16 key research papers on Dense Retrieval 
and Re-ranking techniques. The documentation covers everything from the 
paradigm shift in IR to advanced negative mining strategies.

Features:
• Comprehensive paper collection with links to code
• Theoretical background on hard negatives
• Categorized by technique and year
• Auto-deployed documentation

Link: https://username.github.io/repo-name/
```

**On Reddit (r/MachineLearning):**
```
[P] Curated collection of Dense Retrieval papers with comprehensive documentation

I've created a repository documenting 16 key papers on dense retrieval 
and negative sampling strategies. Includes full explanations of the hard 
negative problem and links to all implementations.

Docs: https://username.github.io/repo-name/
GitHub: https://github.com/username/repo-name
```

---

## 🎉 You're Done!

Your professional documentation is now:
- ✅ Live on the internet
- ✅ Automatically updated on every push
- ✅ Free to host
- ✅ Fully searchable
- ✅ Mobile responsive
- ✅ HTTPS secured

**Congratulations!** 🎊

Now go deploy it! Just follow Steps 1-5 above. You've got this! 💪

---

## 📚 Need Help?

- **GitHub Pages Setup**: See `GITHUB_PAGES_SETUP.md`
- **URL Options**: See `URL_OPTIONS.md`
- **Quick Commands**: See `QUICK_REFERENCE.md`
- **Full Setup Guide**: See `SETUP_COMPLETE.md`

Or just start with Steps 1-5 above - it really is that simple! 🚀

