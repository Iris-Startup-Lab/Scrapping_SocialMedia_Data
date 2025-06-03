# File: E:/font_test/minimal_font_test.py
import fpdf
import fontTools
import os

print(f"fpdf2 version: {fpdf.__version__}")
print(f"fontTools version: {fontTools.__version__}")

# --- Configuration ---
FONT_FAMILY_NAME = 'NotoSans'
FONT_STYLE = ''  # Regular style
TTF_FONT_FILE = 'E:/Users/1167486/Local/fonts/Noto_Sans/static/NotoSans-Regular.ttf'  # Assumed to be in the same directory as this script

# --- Determine Absolute Path to Font File ---
try:
    script_directory = os.path.dirname(os.path.abspath(__file__))
except NameError: 
    script_directory = os.getcwd()
font_file_path = os.path.join(script_directory, TTF_FONT_FILE)

print(f"Attempting to load font: '{TTF_FONT_FILE}'")
print(f"Expected font file location: {font_file_path}")

if not os.path.exists(font_file_path):
    print(f"ERROR: Font file not found at '{font_file_path}'.")
    print(f"Please ensure '{TTF_FONT_FILE}' is in the directory: {script_directory}")
else:
    pdf = fpdf.FPDF()
    try:
        print(f"Calling: pdf.add_font(family='{FONT_FAMILY_NAME}', style='{FONT_STYLE}', fname='{font_file_path}')")
        pdf.add_font(family=FONT_FAMILY_NAME, style=FONT_STYLE, fname=font_file_path)
        print(f"SUCCESS: Successfully added font '{FONT_FAMILY_NAME}' from '{TTF_FONT_FILE}'.")

        pdf.add_page()
        pdf.set_font(FONT_FAMILY_NAME, '', 12)
        pdf.cell(0, 10, "Test with Noto Sans: Hello World € ÄÖÜ ß")
        
        output_pdf_filename = "minimal_test_output.pdf"
        output_pdf_path = os.path.join(script_directory, output_pdf_filename)
        pdf.output(output_pdf_path)
        print(f"SUCCESS: Successfully created PDF: {output_pdf_path}")

    except UnicodeDecodeError as ude:
        print("\n--- UnicodeDecodeError Encountered ---")
        print(f"Error details: {ude}")
        print("This error occurred while fpdf2/fontTools was trying to process the font file's internal metadata.")
        print("This suggests that some text within the font file (like font name, copyright) is not UTF-8 encoded,")
        print("but it was attempted to be read as UTF-8.")
        print("\nTroubleshooting steps taken/suggested:")
        print("1. Ensured fpdf2 and fontTools are updated (check versions printed above).")
        print("2. Used a FRESH, UNCORRUPTED download of 'NotoSans-Regular.ttf' from Google Fonts.")
        print("\nIf this minimal script still fails with these conditions, the issue might be:")
        print("  a) A very specific incompatibility between this font file version and your fontTools/fpdf2 version.")
        print("  b) A deeper environmental issue on your system affecting file reading or encoding interpretation.")
        print("  c) An unusual characteristic of the font file's metadata that triggers a bug.")

    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")