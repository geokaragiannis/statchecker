



class Domain_cell:

    """

    Args
    ====

    -sheet: Sheet object
    -row_index_cell: Cell object of the row index
    -year_cell: Cell object of the year


    Instance Variables
    ==================

    -sheet: Sheet object
    -sheetname: name of sheet
    -file: filename
    -row_index_cell: Cell object of the row index
    -year_cell: Cell object of the year
    -row_index: value of the row index
    -year: vallue of the year
    -domain_coordinate: coordinates of the domain cell
    -domain_cell: Cell object of the domain cell
    -domain_value: value of the domain cell



    """


    
    def __init__(self,sheet,row_index_cell,year_cell):
                    
            
        self.sheet=sheet
        self.sheetname=self.sheet.sheetname

        self.file=self.sheet.file

        self.row_index_cell=row_index_cell
        self.year_cell=year_cell

        self.row_index=self.year=self.domain_coordinate=self.domain_cell=self.domain_value=None
            
            
        if(row_index_cell and year_cell):


            self.row_index=self.sheet.get_cell_value(row_index_cell.coordinate)
            self.year=self.sheet.get_cell_value(year_cell.coordinate)

            self.domain_coordinate=self.year_cell.column_letter+str(self.row_index_cell.row)
            self.domain_cell=self.sheet.get_cell(self.domain_coordinate)

            self.domain_value=self.sheet.get_cell_value(self.domain_coordinate)
        
        
