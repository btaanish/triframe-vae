import PyPDF2
from PyPDF2 import PdfWriter, PdfReader

# Define A4 dimensions in points
A4_WIDTH = 595  # points
A4_HEIGHT = 842  # points

def split_pdf_to_a4(pdf_path, output_dir="."):
    """
    Splits a PDF into 8 A4-sized tiles (2 columns x 4 rows).

    Args:
        pdf_path (str): Path to the input PDF file.
        output_dir (str, optional): Directory to save the output tiles.
    """
    pdf_reader = PdfReader(pdf_path)
    page = pdf_reader.pages[0]  # Assuming only one page in the poster

    width = float(page.mediabox[2])  # Get width from mediabox
    height = float(page.mediabox[3]) # Get height from mediabox

    tile_width = width / 2
    tile_height = height / 4

    for row in range(4):
        for col in range(2):
            output = PdfWriter()

            # Define crop box
            left = col * tile_width
            bottom = height - (row + 1) * tile_height
            right = left + tile_width
            top = bottom + tile_height

            # Create a new blank A4 page
            new_page = page.cropbox
            new_page.lower_left = (left, bottom)
            new_page.upper_right = (right, top)

            # Create a new PDF page
            output.add_page(page)

            output_path = f"{output_dir}/tile_{col}_{row}.pdf"
            with open(output_path, "wb") as output_file:
                output.write(output_file)

            print(f"Created {output_path}")

# Run the function
pdf_path = "CDE2000 Poster 250401.pdf"  #.pdf Replace with your actual PDF path
split_pdf_to_a4(pdf_path)
