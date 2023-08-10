import streamlit as st
import base64

ALLOWED_FILE_TYPES = ["png", "jpg", "jpeg"]
IMAGES_PER_ROW = 3
IMAGE_SCREEN_PERCENTAGE = 1/IMAGES_PER_ROW*100


def main():
    print("Hello world")
    st.set_page_config(page_title="X-Ray Anomalert")
    st.header("X-Ray Anomalert")
    st.text_input("Let's analyze your Chest X-Ray")

    with st.sidebar:
        st.subheader("Your X-Rays")
        uploaded_files = st.file_uploader("Upload your Chest X-Rays and click on 'Process'",
                         accept_multiple_files=True,
                         type=ALLOWED_FILE_TYPES)
        process_button = st.button("Process")

    # Check if the "Process" button is clicked
    if process_button:
        if uploaded_files:
            st.write("Number of X-Rays uploaded:", len(uploaded_files))
            
            # Iterate through the uploaded files and display images in rows
            for i in range(0, len(uploaded_files), IMAGES_PER_ROW):
                row_files = uploaded_files[i:i+IMAGES_PER_ROW]
                
                # Create a row using HTML and CSS styling
                row_html = '<div style="display: flex; ">'
                for file in row_files:
                    # You can perform further processing on each uploaded file here
                    image_data = file.read()
                    image_base64 = base64.b64encode(image_data).decode("utf-8")
                
                    # Use the base64 image data in the img tag
                    row_html += f'<img src="data:image/png;base64,{image_base64}" alt="X-Ray" width="{IMAGE_SCREEN_PERCENTAGE}%" style="margin: 5px;">'
                    # Add text under the image
                    #row_html += f'<p>X-Ray [{i+1}]</p>'  # Change the description accordingly
                row_html += '</div>'
                
                # Display the row
                st.markdown(row_html, unsafe_allow_html=True)
        else:
            st.write("No X-Rays uploaded.")

if __name__ == "__main__":
    main()