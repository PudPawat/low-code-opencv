"""
Function Name: logger.py
*
Description: Create a logger which can fully replace Python built-in function "print"
             See logging.setLoggerClass to design a customized logger
*
Argument: None
*
Parameters: None
*
Return: None
*
Edited by: [2020-10-22] [Bill Gao]
           [2020-11-19] [Bill Gao]
           - Add new function getShowTime

           [2020-12-15] [Bill Gao]
           - Now is able to automatically check and create the log file 
             based on record_path whether it has existed or not

           [2020-12-29] [Bill Gao]
           - stream_enable debug
           - modify the formatter from [Time, Logging_File, Function_Name,  Error_Level, Error_Message]
             to                        [Time, Error_File,   Error_Function, Error_Level, Error_Message]

           [2020-1-5] [Bill Gao]
           - stream & record level debug
           - save every stack message to Error_Message instead of the final one
"""
import logging
import sys
import os
import csv
import traceback
import inspect
from errno import EACCES
from time import strftime


class Log:
    def __init__(self, name:str, stream_level="DEBUG", stream_enable=True, record_level="ERROR", record_path=""):
        """
        Function Name: __init__
        *
        Description: Log is created to replace the built-in function print in Python. Not only
                     it provides very useful information including when does the log happen,
                     happening in which file, or even happening in which subroutine in the file.
                     Also, it provides a extremely practical level system listed in below and
                     are weighted from left to right.
                                [DEBUG, INFO, WARNING, ERROR, CRITICAL]
                                [   10,   20,      30,    40,       50]
                     You can control which log to print and record by setting the stream_level 
                     and the record_level.
        *
        Argument: None
        *
        Parameters: 
                    name [str]           -> name of the logger
                    stream_level [str]   -> the lowest level for the message to be shown,
                                            should be one of the element in [DEBUG, INFO, WARNING, ERROR, CRITICAL]
                    stream_enable [bool] -> enable to print out the message, defaultly True
                    record_level [str]   -> the lowest level for the message to be recorded,
                                            should be one of the element in [DEBUG, INFO, WARNING, ERROR, CRITICAL]
                    record_path [str]    -> record path, should with filename Extension Ex: "setting/log.csv"
        *
        Return: None
        *
        Edited by: [2020-10-23] [Bill Gao]
        """        
        # verify type of name
        assert name != str, "name should be str!"
        
        # setting
        self.__logger = logging.Logger(str(name), level=logging.DEBUG)
        self.__record_enable = bool(record_path)
        self.__stream_enable = bool(stream_enable)
        self.__stream_only_level = 60
        self.__record_only_level = 70    
        self.__stream_default_level, _ = self.__levelCheck(stream_level)
        self.__record_default_level, _ = self.__levelCheck(record_level)
        self.__show_time = "Empty"

        
        # stream console setting                                        
        stream_filter = LevelFilter(filter_name = "stream",
                                    default_level=self.__stream_default_level,
                                    except_level =self.__record_only_level)
        stream_formatter = logging.Formatter("\n[%(show_time)s], [%(error_file)s], [%(error_func)s], [%(level_name)s], \n%(message)s")
        stream_console = logging.StreamHandler()
        stream_console.addFilter(stream_filter)
        stream_console.setFormatter(stream_formatter)
        self.__logger.addHandler(stream_console)

        # record console setting
        if self.__record_enable:
            # check whether csv file exists or is opening
            self.__csvCreater(record_path)

            record_filter = LevelFilter(filter_name = "record",
                                        default_level=self.__record_default_level,
                                        except_level =self.__stream_only_level)
            record_formatter = logging.Formatter("[%(show_time)s], [%(error_file)s], [%(error_func)s], [%(level_name)s], %(message)s")
            record_console = logging.FileHandler(record_path)
            record_console.addFilter(record_filter)
            record_console.setFormatter(record_formatter)
            self.__logger.addHandler(record_console)
            


    def __csvCreater(self, record_path):
        """
        Function Name: __csvCreater
        *
        Description: create csv file at record_path if it doesn't exist, and
                     check whether it is opening
        *
        Argument: None
        *
        Parameters: 
                    record_path [str] -> The path where the file is 
        *
        Return: None
        *
        Edited by: [2020-10-23] [Bill Gao]
        """        
        
        # create a csv file with header if it doesn't exist at the record_path
        log_file_path = os.path.abspath(record_path)
        if os.path.exists(log_file_path) == False:
            if os.path.exists(os.path.dirname(log_file_path)) == False:
                os.mkdir(os.path.dirname(log_file_path))
            with open(record_path,'w', newline="") as file:
                header = ["Time", "Error_File", "Error_Function", "Error_Level", "Error_Message"]
                writedCsv = csv.writer(file)
                writedCsv.writerow(header)  

        # check if csv file is opening or not
        try:
            open_check = open(record_path, "r+") # or "a+", whatever you need
            open_check.close()
        except IOError as io_error:
            if io_error.errno == EACCES:
                error_message = f"File: {record_path} can't be read, please close the file."
                raise IOError(error_message)
            else:
                error_message = f"File: {record_path} can't be read."
                raise IOError(error_message)

    def __levelCheck(self, level):
        """
        Function Name: __levelCheck
        *
        Description: check whether input level fits the format and convert
                     its cooresponding integer
                     [DEBUG,INFO,WARNING,ERROR,CRITICAL] -> [10,20,30,40,50]
        *
        Argument: None
        *
        Parameters: 
                    level [str] -> level of input log      
        *
        Return: 
                [int] -> level of input log
                [str] -> name of the level
        *
        Edited by: [2020-10-23] [Bill Gao]
        """        
        
        # verify type of the log level
        error_message = "Type of log level should be str."
        assert level.__class__ == str, error_message

        # verify the content of level
        if level.upper() == "DEBUG":
            output = logging.DEBUG
        elif level.upper() == "INFO":
            output = logging.INFO
        elif level.upper() == "WARNING":
            output = logging.WARNING
        elif level.upper() == "ERROR":
            output = logging.ERROR
        elif level.upper() == "CRITICAL":
            output = logging.CRITICAL
        else:
            raise ValueError(
                "loglevel should be one of the element in [DEBUG, INFO, WARNING, ERROR, CRITICAL].")
        return output, level.upper()
    
    def __getOutputMessage(self, input_message, name_log_level):
        """
        Function Name: __getOutputMessage
        *
        Description: If the input_message is an error_message, get details from the input_message including
                     error_file, error_func, etc. Otherwise just print it out
        *
        Argument: None
        *
        Parameters: 
                    input_message [type] -> message from the user
                    name_log_level [type] -> name of the log level
        *
        Return: 
                [str] -> output message
                [str] -> names for logging.Formatter
        *
        Edited by: [2021-1-5] [Bill Gao]
        """        
        if input_message.__class__ == str:
            message = inspect.stack()
            if len(message) > 3:
                extra_message = {"show_time": self.__show_time,
                                "error_file": message[-2][1].split("\\")[-1].split(".")[0],
                                "error_func": message[-2][3],
                                "level_name": name_log_level}
                output_message = input_message
            else:
                extra_message = {"show_time": self.__show_time,
                                "error_file": message[-1][1].split("\\")[-1].split(".")[0],
                                "error_func": message[-1][3],
                                "level_name": name_log_level}
                output_message = input_message


        else:
            # checking the type of the input_message
            try:
                input_message.args
            except Exception as error:
                error_message = f"Wrong input type {input_message.__class__} for the input_message, " +\
                                 "it should be either 'str' or the type that has attribute 'args'."
                raise TypeError(error_message)
            output_message = str()

            # get stack messages
            for idx, message in enumerate(inspect.stack()[:1:-1]):
                error_file = message[1]
                line_num = message[2]
                error_func = message[3]
                content_without_space = message[-2][0].strip()
                content = content_without_space.replace("\n", "")
                output_message += f'File "{error_file}", line {line_num}, in {error_func}: {content};'

            # get the last stack message
            detail = input_message.args[0] if any(input_message.args) == True else "Error message is empty"
            error_type = input_message.__class__.__name__
            cl, exc, tb = sys.exc_info()
            last_call_stack = traceback.extract_tb(tb)[-1]
            last_call_file = last_call_stack[0]
            last_call_line = last_call_stack[1]
            last_call_func = last_call_stack[2]
            output_message += f'File "{last_call_file}", line {last_call_line}, in {last_call_func}: [{error_type}] {detail}'
            extra_message = {"show_time": self.__show_time,
                             "error_file": error_file.split("\\")[-1].split(".")[0],
                             "error_func": error_func,
                             "level_name": name_log_level}
            
            # print("\n============ how detail should the logger be? ============")
            # print("### This can see everything from the line that raises an error to where the error comes from ###\n")
            # for idx, call in enumerate(traceback.extract_tb(tb)):
            #     filee = call[0]
            #     line = call[1]
            #     func = call[2]
            #     message = f'File "{filee}", line {line}, in {func}'
            #     print(f"idx: {idx}, message: {message}\n")
            # print(f"detail: {detail}")
            # print("============ how detail should the logger be? ============")

        return output_message, extra_message

    def getShowTime(self):
        """
        Function Name: getShowTime
        *
        Description: This will return the time of the latest calling of function show 
        *
        Argument: None
        *
        Parameters: None
        *
        Return: 
                [str] -> Return the time of the latest calling of function show, or
                         return "Enpty" if there has no recent calling
        *
        Edited by: [2020-11-19] [Bill Gao]
        """        
        return self.__show_time
        
    def show(self, input_message, log_level):
        """
        Function Name: show
        *
        Description: show the input message
        *
        Argument: None
        *
        Parameters: 
                    input_message [str] -> message you want to show
                    log_level [str] -> the lowest level for the message to be shown,
                                       should be one of the element in [DEBUG, INFO, WARNING, ERROR, CRITICAL]
        *
        Return: None
        *
        Edited by: [2020-10-23] [Bill Gao]
        """        

        # verify the content of record_level
        log_level, name_log_level = self.__levelCheck(log_level)
        self.__show_time = strftime("%Y-%m-%d %H:%M:%S")

        # get output message
        output_message, extra_message = self.__getOutputMessage(input_message=input_message,
                                                                name_log_level=name_log_level)

        # show the original output_message and record the revised output_message in .csv file which replaces "," with "-" 
        # to keep the "open file in editor" function in VS Code, also let it suit the format of .csv file
        if output_message.find("File \"") != -1:
            if log_level >= self.__stream_default_level and self.__stream_enable:
                revised_message = output_message.replace(";", "\n")
                self.__logger.log(self.__stream_only_level, revised_message, extra=extra_message)

            if log_level >= self.__record_default_level and self.__record_enable:
                revised_message = output_message.replace(",", " -")
                self.__logger.log(self.__record_only_level, revised_message, extra=extra_message)
        else:
            if log_level >= self.__stream_default_level and self.__stream_enable:
                self.__logger.log(self.__stream_only_level, output_message, extra=extra_message)
                
            if log_level >= self.__record_default_level and self.__record_enable:
                self.__logger.log(self.__record_only_level, output_message, extra=extra_message)



class LevelFilter(logging.Filter):    
    def __init__(self, filter_name, default_level, except_level):
        """
        Function Name: __init__
        *
        Description: If Handler.addFilter is set, every raised log whose level is equal or higher 
                     than Handler.setlevel will be passed through filter.
        *
        Argument: None
        *
        Parameters: 
                    default_level [int] -> should be Handler.setlevel
                    except_level  [int] -> this level should be higher than Handler.setlevel
                                           and would be ignore by LevelFilter
        *
        Return: None
        *
        Edited by: [2020-10-22] [Bill Gao]
        """        
        super().__init__(filter_name)
        self.default_level = default_level
        self.except_level = except_level

    def filter(self, record):
        """
        Function Name: filter
        *
        Description: Filters are consulted in turn by applied logger, if none of them
                     return false, the record will be processed. If one returns false, then
                     no further processing occurs.
                     This is a built_in function of logging.Filter, do not change the name of this
                     subroutine.
        *
        Argument: None
        *
        Parameters: 
                    record [logging.LogRecord] -> message be passed through logger
        *
        Return: [bool] -> True if level of raised log is higher than default_level,
                          and the message will be printed

                          False if level of raised log equals to except_level,
                          and the message will be blocked
        *
        Edited by: [2020-10-22] [Bill Gao]
        """      

        if record.levelno==self.except_level:
            return False
        else:
            return True

