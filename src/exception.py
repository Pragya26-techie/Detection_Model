import sys 
from src.logger import logging


def error_message_deatil(error ,error_detail:sys):
    try:
       _,_,exce_tb = error_detail.exc_info()
       
       if exce_tb is not None:
          file_name = exce_tb.tb_frame.f_code.co_filename 
          line_number = exce_tb.tb_lineno
       else:
           file_number = "Unknown"
           line_number = "Unknoen"
           
       error_message = "Error Occured in python script name [{0}] line number [{1}] eroror message [{2}]".format(
       file_name,exce_tb.tb_lineno,str(error)) 
    
       return error_message
    except Exception as e:
        return f"Error occurred while formatting the original exception: {str(e)} | Original Error: {str(error)}"

class CustomException(Exception):

    def __init__(self , error_message , error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_deatil(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message
    
# if __name__=="__main__":
#     try:
#         a=1/0 
#     except Exception as e:
#         logging.info("Divide by zero error")
#         raise CustomException(e,sys)  