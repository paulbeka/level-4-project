if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

echo "CLONING PROJECT REPO"
git clone https://github.com/paulbeka/level-4-project.git
cd level-4-project

echo "CHECKING IF PYTHON INSTALLED - IF NOT, INSTALL IT"
apt install python3.9
echo "CHECKING IF VENV INSTALLED - IF NOT, INSTALL IT"
apt install -y python3.9-venv
python3 -m venv paul_project_environment
source paul_project_environment/bin/activate
echo "NEW ENVIRONMENT CREATED AND ACTIVE"

pip3 install -r requirements.txt --no-cache-dir
pip3 list

echo "DEPENDANCIES INSTALLED - IF TORCH ERROR CONTACT PAUL"
echo "STARTING NEURAL NETWORK TRAINING - ERROR BEYOND THIS POINT IS PROBABLY GPU RELATED"
cd source/probability_search
python3 score_trainer.py