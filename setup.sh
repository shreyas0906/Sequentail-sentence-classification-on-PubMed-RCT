mkdir -p ~/.streamlit/

mkdir deploy_models
wget https://drive.google.com/file/d/1QvqNFESW9mJohhGvbiUbSqEvfiJckqsB/view
unzip pubmed-model.zip -d deploy_models


echo "\
[server]\n \
headless = true\n\
port=$PORT\n\
enableCORS=false \n\
\n\
" > ~/.streamlit/config.toml
