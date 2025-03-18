"""
Data Loader Module

This module handles loading data from various file formats including:
- CSV
- Excel
- Text files
- PDF
- Images
- JSON
- XML
- Database connections
"""

import os
import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
import docx
import logging
from pathlib import Path
import io
import csv

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.FileHandler("logs/system_logs/data_loader.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class DataLoader:
    """
    Class for loading data from various file formats.
    """
    
    def __init__(self):
        """Initialize the DataLoader class."""
        self.supported_formats = {
            'csv': self._load_csv,
            'xlsx': self._load_excel,
            'xls': self._load_excel,
            'txt': self._load_text,
            'json': self._load_json,
            'xml': self._load_xml,
            'pdf': self._load_pdf,
            'jpg': self._load_image,
            'jpeg': self._load_image,
            'png': self._load_image,
            'docx': self._load_docx,
        }
    
    def load_data(self, file_path):
        """
        Load data from a file.
        
        Args:
            file_path (str): Path to the file.
            
        Returns:
            pandas.DataFrame: Loaded data.
        """
        try:
            # Get file extension
            file_extension = Path(file_path).suffix.lower().replace('.', '')
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check if file format is supported
            if file_extension not in self.supported_formats:
                logger.error(f"Unsupported file format: {file_extension}")
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Load data using the appropriate method
            logger.info(f"Loading data from {file_path}")
            df = self.supported_formats[file_extension](file_path)
            
            # Log success
            logger.info(f"Successfully loaded data from {file_path}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def _load_csv(self, file_path):
        """
        Load data from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file.
            
        Returns:
            pandas.DataFrame: Loaded data.
        """
        try:
            # Try to detect the delimiter
            with open(file_path, 'r', encoding='utf-8') as f:
                sample = f.read(1024)
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample)
                delimiter = dialect.delimiter
            
            # Load the CSV file with the detected delimiter
            df = pd.read_csv(file_path, delimiter=delimiter)
            
            return df
        except Exception as e:
            # If delimiter detection fails, try common delimiters
            for delimiter in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(file_path, delimiter=delimiter)
                    return df
                except:
                    continue
            
            # If all else fails, raise the original exception
            raise e
    
    def _load_excel(self, file_path):
        """
        Load data from an Excel file.
        
        Args:
            file_path (str): Path to the Excel file.
            
        Returns:
            pandas.DataFrame: Loaded data.
        """
        # Read all sheets
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        
        # If there's only one sheet, return it
        if len(sheet_names) == 1:
            return pd.read_excel(file_path, sheet_name=sheet_names[0])
        
        # If there are multiple sheets, return a dictionary of DataFrames
        dfs = {}
        for sheet in sheet_names:
            dfs[sheet] = pd.read_excel(file_path, sheet_name=sheet)
        
        # Combine all sheets into one DataFrame if possible
        # This is a simplification; in a real application, you might want to handle this differently
        main_df = dfs[sheet_names[0]]
        main_df['sheet_name'] = sheet_names[0]
        
        for sheet in sheet_names[1:]:
            temp_df = dfs[sheet]
            temp_df['sheet_name'] = sheet
            
            # Check if the columns are compatible
            if set(main_df.columns) == set(temp_df.columns):
                main_df = pd.concat([main_df, temp_df], ignore_index=True)
        
        return main_df
    
    def _load_text(self, file_path):
        """
        Load data from a text file.
        
        Args:
            file_path (str): Path to the text file.
            
        Returns:
            pandas.DataFrame: Loaded data.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create a DataFrame with the text
        df = pd.DataFrame({'text': [text]})
        
        return df
    
    def _load_json(self, file_path):
        """
        Load data from a JSON file.
        
        Args:
            file_path (str): Path to the JSON file.
            
        Returns:
            pandas.DataFrame: Loaded data.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # If it's a nested dictionary, normalize it
            df = pd.json_normalize(data)
        else:
            df = pd.DataFrame([data])
        
        return df
    
    def _load_xml(self, file_path):
        """
        Load data from an XML file.
        
        Args:
            file_path (str): Path to the XML file.
            
        Returns:
            pandas.DataFrame: Loaded data.
        """
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Convert XML to a list of dictionaries
        data = []
        for child in root:
            item = {}
            for subchild in child:
                item[subchild.tag] = subchild.text
            data.append(item)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        return df
    
    def _load_pdf(self, file_path):
        """
        Load data from a PDF file.
        
        Args:
            file_path (str): Path to the PDF file.
            
        Returns:
            pandas.DataFrame: Loaded data.
        """
        # Extract text using pdfminer
        text = extract_text(file_path)
        
        # Create a DataFrame with the text
        df = pd.DataFrame({'text': [text]})
        
        return df
    
    def _load_image(self, file_path):
        """
        Load data from an image file using OCR.
        
        Args:
            file_path (str): Path to the image file.
            
        Returns:
            pandas.DataFrame: Loaded data.
        """
        # Open the image
        image = Image.open(file_path)
        
        # Extract text using pytesseract
        text = pytesseract.image_to_string(image)
        
        # Create a DataFrame with the text
        df = pd.DataFrame({'text': [text]})
        
        return df
    
    def _load_docx(self, file_path):
        """
        Load data from a Word document.
        
        Args:
            file_path (str): Path to the Word document.
            
        Returns:
            pandas.DataFrame: Loaded data.
        """
        # Open the document
        doc = docx.Document(file_path)
        
        # Extract text
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        
        # Create a DataFrame with the text
        df = pd.DataFrame({'text': [text]})
        
        return df
    
    def load_from_database(self, connection_string, query):
        """
        Load data from a database.
        
        Args:
            connection_string (str): Database connection string.
            query (str): SQL query to execute.
            
        Returns:
            pandas.DataFrame: Loaded data.
        """
        try:
            # Load data using pandas
            df = pd.read_sql(query, connection_string)
            
            return df
        except Exception as e:
            logger.error(f"Error loading data from database: {str(e)}")
            raise 