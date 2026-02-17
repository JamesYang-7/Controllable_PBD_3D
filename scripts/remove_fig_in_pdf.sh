INPUT_FILE="/mnt/c/Users/CGIM/Downloads/pg20251267.pdf"
OUTPUT_FILE="/mnt/c/Users/CGIM/Downloads/pg20251267_nofig.pdf"
gs -sDEVICE=pdfwrite \
   -dFILTERIMAGE \
   -dFILTERVECTOR \
   -dNOPAUSE -dBATCH \
   -sOutputFile=$OUTPUT_FILE \
   $INPUT_FILE