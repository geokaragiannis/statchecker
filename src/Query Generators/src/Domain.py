from src.Domain_cell import Domain_cell


class Domain:

    """

    Args
    ====

    -sheet: Sheet object

    -row_index_year_tuple: tuple of row index and year 
                           Example: ('TPEDToal',2000)


    Instance Variables
    ==================

    -sheet: Sheet object
    -sheetname: name of sheet
    -file: file name
    -row_index: row index of the input tuple
    -year: year of the input tuple
    -row_index_cell: Cell object of the row_index
    -year_cells: list of Cell objects of the year
    -domain_cells: list of Domain_cell objects. It is the intersection of the row of the row_index cell with the column of the year cell
    -domain_coordinates: coordinate of the domain cell
    -domain_values: value of the domain cell 


    """




    
    def __init__(self,sheet,row_index_year_tuple):
        
        
        self.sheet=sheet
        self.sheetname=self.sheet.sheetname
        self.file=self.sheet.file
        self.row_index=row_index_year_tuple[0]
        self.year=row_index_year_tuple[1]
        
        self.row_index_cell=self.sheet.search_row_index_cell(self.row_index)
        self.year_cells=self.sheet.search_year_cell(self.year)
        
        self.domain_cells=[Domain_cell(sheet,self.row_index_cell,x) for x in self.year_cells]
        
        self.domain_coordinates=[x.domain_coordinate for x in self.domain_cells]
        
        self.domain_values=[x.domain_value for x in self.domain_cells]
        
    
        
