import streamlit as st
from fastai.vision.all import load_learner, PILImage
import os

# Load the Fastai model
model_path = 'pokemon_classifier.pkl'
learn_inf = load_learner(model_path)

# Define the Streamlit app
def main():
    st.title("Pokenet: Pokémon Classifier")
    st.write("""
    Welcome to Pokenet! This app uses a machine learning model to classify Pokémon images.
    Please upload an image of a Pokémon, and the model will predict which Pokémon it is.
    """)
    
    # About section
    st.sidebar.title("About")
    st.sidebar.write("""
    Hi! I'm Pratik, a final year BTECH student. I am currently learning deep learning and following Jeremy Howard's lecture on Fastai. This project is made as a practice project and I have to say Fastai has made it so convenient to implement things that might be confusing using PyTorch.

    Connect with me on my socials if you want to discuss DL, ML, DS, or anything.

    [![Twitter](https://img.icons8.com/ios-filled/50/000000/twitter.png)](https://twitter.com/pratik_csv)
    [![LinkedIn](https://img.icons8.com/ios-filled/50/000000/linkedin.png)](https://www.linkedin.com/in/pratik-bokade-b15466230/)
    """)

    # File uploader
    uploaded_file = st.file_uploader("Choose a Pokémon image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Save uploaded file to a temporary directory
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Pokémon image.', use_column_width=True)
        
        # Make prediction
        img = PILImage.create(uploaded_file)
        prediction, _, probs = learn_inf.predict(img)
        
        # Display prediction
        st.success(f"The model predicts this is a {prediction}.")
        st.write(f"Prediction confidence: {probs.max():.2f}")

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    main()
