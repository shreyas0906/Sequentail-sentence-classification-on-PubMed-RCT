import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from spacy.lang.en import English


def split_chars(text):
    """
    Converts a list of string to list of chars
    :param text: string input to be converted to list of chars.
    :return: list of chars.
    """
    return " ".join(list(text))


def preprocess_predict_text(input_text):
    language = English()
    sentencizer = language.create_pipe("sentencizer")
    language.add_pipe(sentencizer)
    doc = language(input_text)
    abstract_lines = [str(sentence) for sentence in list(doc.sents)]

    # Get total number of lines
    total_lines_in_sample = len(abstract_lines)

    # Go through each line in abstract and create a list of dictionaries containing features for each line
    sample_lines = []
    for i, line in enumerate(abstract_lines):
        sample_dict = {"text": str(line), "line_number": i, "total_lines": total_lines_in_sample - 1}
        sample_lines.append(sample_dict)

    test_abstract_line_numbers = [line["line_number"] for line in sample_lines]
    # One-hot encode to same depth as training data, so model accepts right input shape
    test_abstract_line_numbers_one_hot = tf.one_hot(test_abstract_line_numbers, depth=15)

    # Get all total_lines values from sample abstract
    test_abstract_total_lines = [line["total_lines"] for line in sample_lines]

    # One-hot encode to same depth as training data, so model accepts right input shape
    test_abstract_total_lines_one_hot = tf.one_hot(test_abstract_total_lines, depth=20)

    abstract_chars = [split_chars(sentence) for sentence in abstract_lines]

    loaded_model = load_serving_model()
    test_abstract_pred_probs = loaded_model.predict(x=(test_abstract_line_numbers_one_hot,
                                                       test_abstract_total_lines_one_hot,
                                                       tf.constant(abstract_lines),
                                                       tf.constant(abstract_chars)))

    test_abstract_preds = tf.argmax(test_abstract_pred_probs, axis=1)
    label = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']
    test_abstract_pred_classes = [label[i] for i in test_abstract_preds]
    display_to_ui = {"OBJECTIVE": [], "BACKGROUND": [], "CONCLUSIONS": [], "METHODS": [], "RESULTS": []}

    # Visualize abstract lines and predicted sequence labels
    for i, line in enumerate(abstract_lines):
        display_to_ui[test_abstract_pred_classes[i]].append(line)

    return display_to_ui


def load_serving_model():
    serving_model = load_model('deploy_models/model')
    return serving_model


def main():
    page_config = {"page_title": "Sequential sentence classification on PubMed-20k RCT",
                   "page_icon": '📜',
                   'layout': "centered",
                   'initial_sidebar_state': 'auto'}

    st.set_page_config(**page_config)

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
                with st.spinner("Processing..."):
                    ui_elements = preprocess_predict_text(input_text)
                with st.expander("Objective"):
                    if len(ui_elements["OBJECTIVE"]) > 0:
                        for methods in ui_elements["OBJECTIVE"]:
                            st.text(methods)
                with st.expander("Background"):
                    if len(ui_elements["BACKGROUND"]) > 0:
                        for methods in ui_elements["BACKGROUND"]:
                            st.text(methods)
                with st.expander("Methods"):
                    if len(ui_elements["METHODS"]) > 0:
                        for methods in ui_elements["METHODS"]:
                            st.text(methods)
                with st.expander("Results"):
                    if len(ui_elements["RESULTS"]) > 0:
                        for methods in ui_elements["RESULTS"]:
                            st.text(methods)
                with st.expander("Conclusions"):
                    if len(ui_elements["CONCLUSIONS"]) > 0:
                        for methods in ui_elements["CONCLUSIONS"]:
                            st.text(methods)

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
