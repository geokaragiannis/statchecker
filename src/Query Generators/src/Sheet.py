import re
from itertools import islice


"""

Args
====

-file: Book object

-sheetname: names of sheets


Instance Variables
==================

-file: Book object
-sheet: Sheet object
-sheet_val: Sheet object of book_val
-sheetname: name of sheet
-uvn_coordinate: coordinate of unique variable name



Methods
=======

-get_cell: 

 Args: 

-coordinate: coordinate string 

 Returns: Cell of coordinate in sheet


-get_cell_formula:
 Args:
-coordinate: coordinate string
 Returns: formula of cell

-get_cell_value:
 Args:
-coordinate: coordinate string
 Returns: value of cell


-find_UVN_cell:

Returns: cell of unique variable identifier

###############################
# Check code about counting _ #
###############################


-search_row_index_cell:
 Args:
-row_index: row index value
 Returns: cell of row index 

-search_year_cell:
 Args:
-year: year value
 Returns: list of cells of year 


"""




class Sheet:
    
    uvn_titles=['Unique Variable Name','Lokkup Name','Lookup','Vlookup name']
    
    
    def __init__(self,file,sheetname):
        
        self.file=file.file
        self.sheet=file.book[sheetname]
        self.sheet_val=file.book_val[sheetname]
        self.sheetname=sheetname
        self.uvn_coordinate=None
        
        
        
        
    @staticmethod
    def check_valid_coordinate(coordinate):
        if(re.match(r"[A-Z]+\d+",coordinate)):
            return True
        return False
        
        

    def get_cell(self,coordinate):
        
        if(not self.check_valid_coordinate(coordinate)):
            print("{} is an invalid coordinate.".format(coordinate))
            return
        
        return self.sheet[coordinate]
        
        
        
    def get_cell_formula(self,coordinate):
        
        if(not self.check_valid_coordinate(coordinate)):
            print("{} is an invalid coordinate.".format(coordinate))
            return
        
        return self.sheet[coordinate].value
        
        
    def get_cell_value(self,coordinate):
        
        if(not self.check_valid_coordinate(coordinate)):
            print("{} is an invalid coordinate.".format(coordinate))
            return
        
        return self.sheet_val[coordinate].value
    
    
    
    def find_UVN_cell(self):
        
        if(self.get_cell_value('D4') in Sheet.uvn_titles):
            self.uvn_coordinate=self.sheet['D4']
        
        if(self.get_cell_value('C5') in Sheet.uvn_titles):
            self.uvn_coordinate=self.sheet['C5']
            
        if(self.get_cell_value('C4') in Sheet.uvn_titles):
            self.uvn_coordinate=self.sheet['C4']
       # if(self.uvn_coordinate):
       #     print("Found UVN cell {}".format(self.uvn_coordinate.coordinate))
            
            

    def search_row_index_cell(self,row_index):
        if(not self.uvn_coordinate):
            self.find_UVN_cell()
        if(self.uvn_coordinate):
            self.uvn_column=self.uvn_coordinate.column_letter
            self.uvn_row=int(self.uvn_coordinate.row)
            
            
            for i in range(self.uvn_row,self.sheet.max_row+1):
                cell_coordinate=self.uvn_column+str(i)
                if(self.sheet[cell_coordinate].value==row_index):
                    return self.sheet[cell_coordinate]
                    
           # print("Didn't find row index")
            return None
        #print("You have to implement this.")
        return None
        
        
        
    def search_year_cell(self,year):
        year_cells=[]
        n=10
        first_n_rows=islice(self.sheet_val.rows,n)
        
        for row in first_n_rows:
            for cell in row:
                if(cell.value==year and cell.column<94):
                    year_cells.append(cell)
        return year_cells
    
    
    def build_table(self,row_indices,years):
        pass
    
    def get_domains(self,row_indices,years):
        pass
