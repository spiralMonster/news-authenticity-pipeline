import streamlit as st
from rich import layout

from making_prediction_from_model_server.via_rest_api.making_predictions_via_rest_api import MakePredictions
from collect_feedback import CollectFeedback


st.set_page_config(
    page_title="News Authentication",
    layout="centered",
)


if "prediction_done" not in st.session_state:
    st.session_state.prediction_done=False

if "result" not in st.session_state:
    st.session_state.result=""

if "feedback" not in st.session_state:
    st.session_state.feedback=""

if "news_text" not in st.session_state:
    st.session_state.news_text=""


st.markdown(
    """
    <h1 style='text-align: center;'>
        News Authentication
    </h1>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.write("")

st.session_state.news_text=st.text_area(
    "Enter news",
    height=200,
    placeholder="Paste the news article here..."
)

st.write("")

if not st.session_state.prediction_done:
    button=st.button(
        "Check authenticity",
        use_container_width=True,
        type="primary",
    )

    if button:
        if len(st.session_state.news_text.strip())==0:
            st.warning("Please enter some news text.")

        else:
            response=MakePredictions(text=st.session_state.news_text)["predictions"][0][0]
            print(response)

            if response<0.5:
                news_authenticity="Fake News"

            else:
                news_authenticity="True News"


            st.session_state.result=news_authenticity
            st.session_state.prediction_done=True

            st.rerun()


if st.session_state.prediction_done:
    st.markdown("### Result")

    st.success(st.session_state.result)
    st.write("")

    st.markdown("### Was this prediction correct?")

    col1,col2=st.columns(2)

    pred=st.session_state.result
    pred=pred.split(" ")[0]
    prediction_made=pred.lower()

    with col1:
        if st.button("Yes"):
            st.success("Thanks for your feedback!")

            st.session_state.feedback=prediction_made

            CollectFeedback(
                text=st.session_state.news_text,
                label=st.session_state.feedback
            )


    with col2:
        if st.button("No"):
            st.info("Thanks! We'll use your feedback to improve.")

            if prediction_made=="true":
                feedback="fake"

            elif prediction_made=="fake":
                feedback="true"

            st.session_state.feedback=feedback
            CollectFeedback(
                text=st.session_state.news_text,
                label=st.session_state.feedback
            )


    st.write("")
    st.write("")
    c1,c2,c3=st.columns(3)

    with c2:
        check_another_button = st.button("Check Another News")
        if check_another_button:
            st.session_state.prediction_done=False
            st.session_state.result=""
            st.session_state.feedback=""
            st.session_state.news_text=""

            st.rerun()

