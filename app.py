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
    """

    Args:
        input_text: unprocessed raw text from the user.
        Needs to be tokenized and in the same format as the trained model.

    Returns: processed text with line numbers
    """
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
    print("loading model...")
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

        st.subheader("Sample Abstracts")
        st.text("Hepatitis C virus (HCV) and alcoholic liver disease (ALD), either alone or in combination,\n "
                "count for more than two thirds of all liver diseases in the Western world. There is no safe level of\n "
                "drinking in HCV-infected patients and the most effective goal for these patients is total \n"
                "abstinence. Baclofen, a GABA(B) receptor agonist, represents a promising pharmacotherapy for alcohol\n "
                "dependence (AD). Previously, we performed a randomized clinical trial (RCT), which demonstrated the\n "
                "safety and efficacy of baclofen in patients affected by AD and cirrhosis. The goal of this post-hoc\n "
                "analysis was to explore baclofen's effect in a subgroup of alcohol-dependent HCV-infected cirrhotic\n "
                "patients. Any patient with HCV infection was selected for this analysis. Among the 84 subjects\n "
                "randomized in the main trial, 24 alcohol-dependent cirrhotic patients had a HCV infection; 12\n "
                "received baclofen 10mg t.i.d. and 12 received placebo for 12-weeks. With respect to the placebo\n "
                "group (3/12, 25.0%), a significantly higher number of patients who achieved and maintained total\n "
                "alcohol abstinence was found in the baclofen group (10/12, 83.3%; p=0.0123). Furthermore,\n "
                "in the baclofen group, compared to placebo, there was a significantly higher increase in albumin\n "
                "values from baseline (p=0.0132) and a trend toward a significant reduction in INR levels from\n "
                "baseline (p=0.0716). In conclusion, baclofen was safe and significantly more effective than placebo\n "
                "in promoting alcohol abstinence, and improving some Liver Function Tests (LFTs) (i.e. albumin,\n "
                "INR) in alcohol-dependent HCV-infected cirrhotic patients. Baclofen may represent a clinically \n"
                "relevant alcohol pharmacotherapy for these patients.\n")

        st.text("Mental illness, including depression, anxiety and bipolar disorder, accounts for a significant\n "
                "proportion of global disability and poses a substantial social, economic and heath burden. Treatment\n "
                "is presently dominated by pharmacotherapy, such as antidepressants, and psychotherapy,\n "
                "such as cognitive behavioural therapy; however, such treatments avert less than half of the disease\n "
                "burden, suggesting that additional strategies are needed to prevent and treat mental disorders.\n "
                "There are now consistent mechanistic, observational and interventional data to suggest diet quality\n "
                "may be a modifiable risk factor for mental illness. This review provides an overview of the\n "
                "nutritional psychiatry field. It includes a discussion of the neurobiological mechanisms likely\n "
                "modulated by diet, the use of dietary and nutraceutical interventions in mental disorders,\n "
                "and recommendations for further research. Potential biological pathways related to mental disorders\n "
                "include inflammation, oxidative stress, the gut microbiome, epigenetic modifications and\n "
                "neuroplasticity. Consistent epidemiological evidence, particularly for depression, suggests an\n "
                "association between measures of diet quality and mental health, across multiple populations and age\n "
                "groups; these do not appear to be explained by other demographic, lifestyle factors or reverse\n "
                "causality. Our recently published intervention trial provides preliminary clinical evidence that\n "
                "dietary interventions in clinically diagnosed populations are feasible and can provide significant\n "
                "clinical benefit. Furthermore, nutraceuticals including n-3 fatty acids, folate,\n "
                "S-adenosylmethionine, N-acetyl cysteine and probiotics, among others, are promising avenues for\n "
                "future research. Continued research is now required to investigate the efficacy of intervention\n "
                "studies in large cohorts and within clinically relevant populations, particularly in patients with\n "
                "schizophrenia, bipolar and anxiety disorders.\n")

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
