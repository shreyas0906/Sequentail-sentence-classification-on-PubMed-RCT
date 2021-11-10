import streamlit as st


def main():
    PAGE_CONFIG = {"page_title": "Sequential sentence classification on PubMed-20k RCT",
                   "page_icon": '📜',
                   'layout': "centered",
                   'initial_sidebar_state': 'auto'}

    st.set_page_config(**PAGE_CONFIG)

    menu = ["Home", "Architecture and details", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.header("Sequential sentence classification on PubMed-20k RCT")
        st.markdown(
            "In this application, given an input abstract text from the user, the deep learning model will classify "
            "each "
            "sentence into the following categories \n"
            "-  METHODS\n"
            "-  RESULTS\n"
            "-  CONCLUSIONS\n"
            "-  BACKGROUND\n"
            "-  OBJECTIVE\n")

        input_text = st.text_area("Enter abstract here")
        if st.button("Submit"):
            if len(input_text) > 0:
                '''
                pass the text to predict the sentences.
                '''
                with st.expander("Methods"):
                    pass  # st.text() # pass the output methods here
                with st.expander("Results"):
                    pass
                with st.expander("Conclusions"):
                    pass
                with st.expander("Background"):
                    pass
                with st.expander("Objective"):
                    pass
            else:
                st.error("No text data given")

    if choice == "Architecture and details":
        st.image('tribrid.png', caption="model architecture with character embeddings, token embeddings and "
                                        "positional embeddings.", use_column_width='auto')
        # st.text() output model summary()
        # Print model training process

    if choice == "About":
        st.subheader("Sequential-sentence-classification-on-PubMed-RCT")
        st.write("In this repo, we'll be doing sequential sentence classification using token, character and "
                 "positional embeddings based on this "
                 "[paper](https://arxiv.org/abs/1710.06071). "
                 "To use this repo, please clone this [repo](https://github.com/Franck-Dernoncourt/pubmed-rct) "
                 "and place it in the same repo as this repo as shown in the figure below.")
        st.image('directory_structure.png', caption="directory structure")
        st.write("The structure of the project is as follows.\n"
                 "- src/models.py\n"
                 "   - contains all the different network architectures to run experiments on.\n"
                 "- src/data.py\n"
                 "   - contains all the necessary pre-processing actions that need to be carried out to convert the dataset\n"
                 "into a tf.data.Dataset pipeline for the model.\n"
                 "- src/callbacks.py\n"
                 "   - all the callbacks necessary for model training and monitoring is located here.\n"
                 "- train.py\n"
                 "   - uses all the above mentioned files to start training the model.\n"
                 "usage:\n")
        st.code("python3 train.py --epochs 25 --train True", language='python')
        st.write("\nEverything else is pretty much self-explanatory.\n"
                 "\nIf you have any questions, please feel free to contact me at shreyas0906@gmail.com or raise an issue and i will have a"
                 "look into it.")


if __name__ == '__main__':
    main()
