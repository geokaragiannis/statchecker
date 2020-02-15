import pandas as pd
import sqlite3



class TableBuilder:
    
    def __init__(self,bfc):
        self.bfc=bfc
        self.mini_table=None
        self.solutions=list(set([xx for x in self.bfc.solutions for xx in x]))
    def build_table(self):
        
      #  df_tuples=list(set([(x.row_index,x.year,x.domain_value) for x in self.sfc.domain_cells if x.domain_value]))
        df_tuples=list(set([(x.row_index,x.year,x.domain_value) for x in self.solutions if x.domain_value]))
        df_tuples=sorted(df_tuples)




        row_indices=list(set([x[0] for x in df_tuples]))
        year_columns=sorted(list(set(['Y'+str(x[1]) for x in df_tuples])))
        memory_tabs_cols=['RowIndex']+year_columns
        a=pd.DataFrame(columns=memory_tabs_cols,index=row_indices)

        for x in df_tuples:
            a.loc[x[0]]['Y'+str(x[1])]=x[2]

        a['RowIndex']=a.index
        
        self.mini_table=a
        
    def query_table(self,query):
        
        
        try:
            
            #build table in memory
            conn = sqlite3.connect(':memory:')
            
            #add UDFs
            conn.create_function("POWER", 2, pow)
            
            self.mini_table.to_sql('df',conn)
            
            c = conn.cursor()
            c.execute(query)
            return c.fetchall()[0][0]
        
        
        except sqlite3.Error as error:
            print("Error while connecting to sqlite",error)

        finally:
            if(conn):
                conn.close()

