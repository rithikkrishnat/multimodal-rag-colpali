import os
from pdf2image import convert_from_path
from PIL import Image

# Your exact Poppler path from the screenshot
POPPLER_PATH = r"C:\poppler-25.12.0\Library\bin"

def pdf_to_images(pdf_path, output_folder):
    """
    Converts a PDF into high-resolution images for ColPali processing.
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    print(f"Processing '{pdf_path}'...")
    print("Rasterizing pages (this might take a moment)...")
    
    # Convert PDF to a list of PIL Images
    # DPI=200 is a good balance between quality and model memory usage
    try:
        pages = convert_from_path(
            pdf_path, 
            dpi=200, 
            poppler_path=POPPLER_PATH 
        )
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []
    
    saved_paths = []
    
    # Save each page as a JPEG
    for i, page in enumerate(pages):
        # We save as RGB JPEG to keep file sizes manageable
        page = page.convert('RGB')
        
        filename = f"page_{i + 1}.jpg"
        save_path = os.path.join(output_folder, filename)
        
        page.save(save_path, "JPEG")
        saved_paths.append(save_path)
        print(f"Saved: {save_path}")
        
    print(f"Successfully converted {len(pages)} pages.")
    return saved_paths

if __name__ == "__main__":
    # Test it out! 
    # 1. Put a sample PDF named 'sample_report.pdf' in your colpali folder
    sample_pdf = "sample_report.pdf" 
    output_dir = "processed_images"
    
    if os.path.exists(sample_pdf):
        images = pdf_to_images(sample_pdf, output_dir)
    else:
        print(f"Please place a file named '{sample_pdf}' in this directory to test.")