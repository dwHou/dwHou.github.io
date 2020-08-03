Command line instructions

Git global setup

git config --global user.name "侯德旺"
git config --global user.email "dewanghou@gmail.com"

Create a new repository

git clone git@code.company.org:videoarch/dw.git
cd dw
touch README.md
git add README.md
git commit -m "add README"
git push -u origin master

Existing folder

cd existing_folder
git init
git remote add origin git@code.company.org:videoarch/dw.git
git add .
git commit -m "Initial commit"
git push -u origin master

Existing Git repository

cd existing_repo
git remote rename origin old-origin
git remote add origin git@code.company.org:videoarch/dw.git
git push -u origin --all
git push -u origin --tags
