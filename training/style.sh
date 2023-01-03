#!/usr/bin/env bash

if [ ! -d "stylegan" ]; then
  echo -e "\x1b[31mDirectory doesnt exist!\x1b[0m";
  read -r -p $'\e[36mWould you like to clone the repository? \e[0;37m(y/n)\e[0m ' decision;
  if [ "$decision" = "y" ]; then
    echo -e "\x1b[33mMaking Directory\x1b[0m";
    mkdir "stylegan";

    git clone "https://github.com/NVlabs/stylegan2-ada-pytorch" "stylegan"
    echo -e "\x1b[32mDirectory created!\x1b[0m"
    exit 0
  else
    echo -e "\x1b[31mExiting...\x1b[0m";
    exit 1;
  fi
fi

echo -e "\x1b[33mStarting training...\x1b[0m"
read -r -p $'\e[36mDry Run? \e[37m(y/n)\e[0m ' dry;
if [ "$dry" = "y" ]; then
  dry="--dry-run"
else
  dry=""
fi

if [ ! -f "data/images.zip" ]; then
  python stylegan/dataset_tool.py --source=data/images --dest=data/images.zip --width=256 --height=256
  echo -e "\x1b[32mDataset created!\x1b[0m"
fi
python stylegan/train.py --outdir="stylegan_results" --data="data/images.zip" --cond="True" "$dry"

