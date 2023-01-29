git clone https://github.com/paulbeka/level-4-project.git
cd level-4-project

[[ "$(python3 -V)" =~ "Python 3" ]] && echo "Python 3 is installed"

python3 -m venv paul_project_environment
source paul_project_environment/bin/activate

pip install -r requirements.txt

cd source/probability_search
python3 grid_probability_trainer.py