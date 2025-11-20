mkdir -p /workspace/reward_seeker;
exec > /workspace/onstart.log 2>&1;
set -ex;

apt-get update;
apt-get install -y git screen python3-venv;

mkdir -p /root/.config/chezmoi;
cat <<EOF > /root/.config/chezmoi/chezmoi.toml
encryption = "age"
[git]
autoCommit = true
autoPush = true
[age]
identity = "/root/key.txt"
recipient = "age1vj6r9tjp5k39mn4fhf55qja6gjncgljn6zjuw0656qlyzdh7ysks5ndefg"

EOF
 
cat <<EOF > /root/key.txt
AGE-SECRET-KEY-1CUZLEJ2JY8VKCCC4Z6Z5RS0EZKK3ZW4XED2S4Z69KSAZWTKZPTCQ8SALEL
EOF
 
sh -c "$(curl -fsLS get.chezmoi.io/lb)" -- init --apply aghyad-deeb;
 
chmod 600 /root/.ssh/config || true;

chmod 600 ~/.ssh/config; rm -rf ~/.local/share/chezmoi; sh -c "$(curl -fsLS get.chezmoi.io/lb)" -- init --apply git@github.com:aghyad-deeb/dotfiles.git
 
git clone git@github.com:aghyad-deeb/reward_seeker.git /workspace/reward_seeker;
mv /workspace/reward_seeker/gitignore /workspace/reward_seeker/.gitignore
git lfs install

 
curl -LsSf https://astral.sh/uv/install.sh | sh;
 
python3 -m venv /workspace/reward_seeker/venv;
 
/root/.local/bin/uv pip install -r /workspace/reward_seeker/requirements.txt --python /workspace/reward_seeker/venv/bin/python;
/root/.local/bin/uv pip install numpy==2.2 --python /workspace/reward_seeker/venv/bin/python;

source /workspace/reward_seeker/venv/bin/activate;
mkdir -p /workspace/reward_seeker/models;
wandb login "15b1216ae957676be6cbbd1afba25f920ce1c938";
 
echo "Onstart script completed successfully.";
git config --global user.email "th3elctronicag@gmail.com";
git config --global user.name "aghyad-deeb";
cd /workspace/reward_seeker;
git lfs track data;
git lfs track **/*.log;
git config pull.rebase false
git lfs track *.parquet;
git lfs track *.jsonl;
git lfs track *.json;
sudo ln -sf /usr/share/zoneinfo/America/Los_Angeles /etc/localtime

