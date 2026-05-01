import sys

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    return f"Error occurred in script [{exc_tb.tb_frame.f_code.co_filename}] at line [{exc_tb.tb_lineno}] : {str(error)}"

class CustomException(Exception):
    def __init__(self, error, error_detail: sys):
        super().__init__(error_message_detail(error, error_detail))