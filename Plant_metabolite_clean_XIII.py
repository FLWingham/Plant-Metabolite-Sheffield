"""
dataset == individual text file (which includes the i,j,k versions). Example: UVF2CNa1ai\ UVF2CNa1aj\ UVF2CNa1ak
complete_dataset == all filtered or raw data files contains every dataset. Example: AlpA1\plants
"""
#===================================================================================================================================#

import numpy as np
from itertools import tee, islice, izip, groupby
from operator import itemgetter
import heapq
import os, glob, sys
import errno
import bisect
import math
import itertools
from pylab import plot,subplot,axis,stem,show,figure
from numpy import mean,cov,cumsum,dot,linalg,size,flipud,argsort

#===================================================================================================================================#

rootdir = os.getcwd()
intro_message = True
extension = [ "i.txt","j.txt","k.txt"]

#===================================================================================================================================#

def flatten(l):
  out = []
  for item in l:
    if isinstance(item, (list, tuple)):
      out.extend(flatten(item))
    else:
      out.append(item)
  return out

#===================================================================================================================================#

def save(list_to_save,file_name,target_directory):
    target = os.path.join(target_directory,file_name+".txt")
    np.savetxt(target,list_to_save)

#===================================================================================================================================#

def remove_raw_extension(s):
    badSubstrings = ["_raw"]
    for badSubstring in badSubstrings:
        s = s.replace(badSubstring, "")
    return s

#===================================================================================================================================#

def create_directory(current_directory,directory_name):
    new_directory_name = remove_raw_extension(directory_name)
    current_path = os.path.join(rootdir,new_directory_name)
    path = remove_raw_extension(current_path)

    try:     
        os.makedirs(path)

    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    return {'path':path}

#===================================================================================================================================#

def existence_check(current_directory,file_name):
    #returns true iff file exists elif == false => file doesn't existence_check
    file_path = os.path.join(current_directory,file_name)
    return os.path.exists(file_path)


#===================================================================================================================================#

def query_yes_no(question, default=None):

    valid = {"yes":True,   "y":True,  "ye":True, "Yes":True, "yy":True ,
             "no":False,  "nn":False,   "n":False, "No":False}
    if default == None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "\
                             "(or 'y' or 'n').\n")
     
#===================================================================================================================================#

"""
def ijk_pattern_exists(list1, list2):
    return ''.join(map(str, list2)) in ''.join(map(str, list1))
""" 
#===================================================================================================================================#

def cut_extension(s,n):    
    return s[:n]

#===================================================================================================================================#

def remove_extensions(s):
    badSubstrings = ["i.txt", "j.txt", "k.txt"]
    for badSubstring in badSubstrings:
        s = s.replace(badSubstring, "")
    return s
  
#===================================================================================================================================#

def user_identifier(input_list):
    input_list_user= [input_list[n]+' = '+str(n+1) for n in range(len(input_list))]
    return input_list_user

#===================================================================================================================================#

"""
def pretty_print(CL_output):
    columns = len(CL_output)//200+2
    lines = ("".join(s.ljust(20) for s in CL_output[i:i+columns-1])+CL_output[i:i+columns][-1] for i in range(0, len(CL_output), columns))
    print "\n".join(lines)
"""

#===================================================================================================================================#

def pretty_print(CL_output):
    if len(CL_output)%2 == 0:
        #even
        print "\n".join("%-20s %s"%(CL_output[i],CL_output[i+len(CL_output)/2]) for i in range(len(CL_output)/2))    
    else:
        #odd
        d_odd = CL_output + ['']
        print "\n".join("%-20s %s"%(d_odd[i],d_odd[i+len(d_odd)/2]) for i in range(len(d_odd)/2))


#===================================================================================================================================#
   
def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if x not in seen and not seen_add(x)]

#===================================================================================================================================#

def txtfile_cleanup(file_names):

    file_names_extension_remove = [remove_extensions(n) for n in file_names]
    file_names_no_dup = remove_duplicates(file_names_extension_remove)
    pretty_print(user_identifier(file_names_no_dup))

    return file_names_no_dup

#===================================================================================================================================#

def dir_walk(path): 
    all_dirs =  os.walk(path).next()    
    return {'py_path':all_dirs[0],'level_0_dirs':all_dirs[1],'file_names':all_dirs[2]}

#===================================================================================================================================#

def print_current_dir(directory_path):

    target_path = os.path.join(rootdir,directory_path)

    all_dirs = dir_walk(target_path)

    d_dirs = all_dirs['level_0_dirs']
    d_files = all_dirs['file_names']
    d_files_txt =  [x for x in d_files if '.txt' in x]
    
    if d_dirs:
        pretty_print(user_identifier(d_dirs))
        return d_dirs

    elif not d_dirs:
        clean_txt_files = txtfile_cleanup(d_files_txt)
        return d_files

#===================================================================================================================================#

def print_filtered_or_PCA_directories(directory_path, PCA):
    
    target_path = os.path.join(rootdir,directory_path)

    all_dirs = dir_walk(target_path)

    d_dirs = all_dirs['level_0_dirs']

    if PCA == True:
      d_filtered_files = [x for x in d_dirs if not '_raw' in x]
      
    elif PCA == False:
      d_filtered_files = [x for x in d_dirs if '_raw' in x]

    pretty_print(user_identifier(d_filtered_files))
    return d_filtered_files

#===================================================================================================================================#

def index_containing_substring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
              return i
    return -1

#===================================================================================================================================#

def open_file(directory, filename):

    loaded_data = []
 
    path = os.path.join(directory,filename)

    data = np.loadtxt(path)

    loaded_data.append(data)

    return np.array(loaded_data)

#===================================================================================================================================#
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
# BEGIN PLANT METABOLITE SCRIPT
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
#===================================================================================================================================#

def Plant_metabolite_python_script():

#===================================================================================================================================#
# BEGIN FILTER
#===================================================================================================================================#

    def sort_and_group(data_input):
        
        data = [[data_input[n][m].tolist()+ [n] for m in range(len(data_input[n]))] for n in range(len(data_input))]

        data_flat = np.concatenate(data).tolist()
        
        data_sorted = sorted(data_flat, key = itemgetter(0))

        groups = list(izip(*[islice(el, n, None) for n, el in enumerate(tee(data_sorted, 3))]))

        return groups

    #===============================================================================================================================#

    def filter(dataset):

        data_clean_mz = []
        data_clean_tic = []

        a = 0.00003
        b = 0.0033

        groups = sort_and_group(dataset)

        for n in range(len(groups)):

            s   = [groups[n][0][2],groups[n][1][2],groups[n][2][2]]
            
            counter =[s.count(0.0), s.count(1.0),s.count(2.0)]

            if counter == [1,1,1]:
                mz = [groups[n][0][0],groups[n][1][0],groups[n][2][0]]
                tic = [groups[n][0][1],groups[n][1][1],groups[n][2][1]]
                
                ESminus = a*np.mean(mz)+ b
                sigma = np.std(mz, ddof = 1)
                
                if sigma < ESminus:
                    data_clean_mz.extend(mz)
                    data_clean_tic.extend(tic)
                else:
                    pass

        clean_data = np.array(zip(data_clean_mz, data_clean_tic))

        return {'clean_data': clean_data,'mz': data_clean_mz, 'tic': data_clean_tic}

#===================================================================================================================================#
# END FILTER
#===================================================================================================================================#
# BEGIN CALIBRATE
#===================================================================================================================================#

    def calibrate(ref_dataset,cal_dataset):

        no_peaks = 5000
        polynomial_order = 1
        mz_range = 0.125
        Di = ref_dataset[0]
        Dj = cal_dataset 
        
        Ref =  np.asarray(heapq.nlargest(no_peaks,Di, key = itemgetter(1)))
        Cal =  np.asarray(heapq.nlargest(no_peaks,Dj, key = itemgetter(1)))
        
        order = np.argsort(Cal[:, 0])
        bx = Cal[order, 0]

        sidx = np.searchsorted(bx, Ref[:, 0] - mz_range, side="right")
        eidx = np.searchsorted(bx, Ref[:, 0] + mz_range, side="left")
        
        result = []; Reference_mz = []; calibrated_mz = []; Ratio = []

        for s, e, Ref_tic, n in zip(sidx, eidx, Ref[:, 1], range(len(Ref))):
     
            section = order[s:e]
           
            
            if section.size != 0:
                Reference_mz.append(Ref[n,0])
                Cal_tic = Cal[section, 1]
                idx = np.argmin(np.abs(Ref_tic-Cal_tic))
                result.append(Cal[section[idx]])
            else:
                pass
            
        result = np.array(result)

        Cal_mz = result[:,0]
        Cal_tic = result[:,1]

        Offset = [i-j for i,j, in zip(result[:,0], Reference_mz)]

        no_peaks_within_mz = len(result)

        LBF = np.polyfit(Reference_mz,Offset,polynomial_order)                    


        for n in range(no_peaks_within_mz):

            f = LBF[1] + LBF[0]*(Reference_mz[n])
           
            Residual = Offset[n] - f
            R = np.abs(Residual/np.std(Offset,dtype=np.longdouble))
            Ratio.append(R)

        Offset_sigma= [offset for offset, ratio in zip(Offset, Ratio) if ratio < 2.5]
        Ref_mz_sigma = [ref_mz for ref_mz, ratio in zip(Reference_mz,Ratio) if ratio < 2.5]
        Cal_mz_sigma = [cal_mz for cal_mz, ratio in zip(Cal_mz,Ratio) if ratio < 2.5]
    
        LBF_sigma = np.polyfit(Ref_mz_sigma,Offset_sigma,polynomial_order)
     
        no_outliers = len(Reference_mz)-len(Ref_mz_sigma)

        for n in range(len(cal_dataset)):
            Cmz = Dj[n][0] + LBF_sigma[1] + LBF_sigma[0]*(Dj[n][0]) 
            calibrated_mz.append(Cmz)

        calibrated_tic = Dj[:,1]

        calibrated_data = np.array(zip(calibrated_mz,calibrated_tic))
        
        return {'calibrated_data':calibrated_data,'cal_mz':calibrated_mz,'cal_tic':calibrated_tic}

#===================================================================================================================================#
# END CALIBRATE
#===================================================================================================================================#
# BEGIN PCA
#===================================================================================================================================#

    def drange(start, stop, step):
         r = start
         while r < stop:
            yield r
            r += step

    #===============================================================================================================================#

    def binning(complete_dataset):

        output = []

        for data in complete_dataset:

            bin_location = []

            mz_data = [n[0] for n in data if 50<n[0]<600]
            
            tic_data = [n[1] for n in data if 50<n[0]<600]
            
            bins = list(drange(0.5,600.5,1))
            
            for mz,tic in zip(mz_data,tic_data):

                loc = bisect.bisect_left(bins, mz) - 1
                bin_location.append(loc)

           
            p = zip(tic_data,bin_location)
          
            data_insert_zeros = [k for i,j in enumerate(p,1) for k in [j]
                         + [(0,n) for n in range(j[1]+1,p[min(i,len(p)-1)][1])]]  # p[min(i,len(p)-1)][1] is replaced by len(bins). #this code will enter a (0,n) whenever there is a mass dimension at loction n which is empty.
            
            output_unequal_len = [zip(sum([x for x, y in group])
                      for key, group in groupby(data_insert_zeros, key=itemgetter(1)))][0] # this code groups tic values in the same mass bin and sums.  

            reformatted_output = list(itertools.chain(*[list(n) for n in output_unequal_len]))

            output.append(reformatted_output)
            
        maximum = [len(bins)-len(n) for n in output]
        
        binned_data = np.array([n+[0]*m for n,m in zip(output,maximum)])

        return binned_data
   
    #===============================================================================================================================#

    def princomp(A,numpc=0):

        [obs, dims] = np.shape(A)
    
        mu = A.mean(axis=0)
        std = np.std(A, ddof=0)
     
        A = (A-mu)/np.tile(std, (obs,1))

        C = np.cov(A.T, rowvar=0, ddof=1)
        
        value, vector = np.linalg.eig(C)

        p = size(vector,axis=1)
        
        idx = argsort(value) # sorting the eigenvalues
        
        idx = idx[::-1]       # in ascending order
       
        vector = vector[:,idx] # sorting eigenvectors according to the sorted eigenvalues
        
        value = value[idx] # sorting eigenvalues
  
        explained_variance = [n*100/sum(value) for n in value] # given as a percentage of explained variance

        var_val = zip(explained_variance, value)
    
        if numpc < p and numpc >= 0:
            vector = vector[:,range(numpc)] # cutting some PCs if needed

        projected_data = dot(vector.T,A)
  
        return {'projected_data':projected_data, 'explained_variance':explained_variance, 'eigenvalues':value, 'variance_value':var_val}
       
        
#===================================================================================================================================#
# END BINNING
#===================================================================================================================================#
# BEGIN COMMAND LINE INTERPRETER (CLI)
#===================================================================================================================================#

    def initial_user_input(structure_type):

        if structure_type == "directory":

            mydict ={ "initial_message":"Please enter a number to open the corresponding directory: ",
            }
            correct_response = False
            while correct_response == False:
                try:
                    print ''
                    user_input = int(raw_input(mydict["initial_message"]))
                    print ''
                    return user_input
                    correct_respond = True
                except ValueError:
                    print 'Incorrect input. Please enter a number to open the corresponding directory. '     

        #============================================================================================#

        elif structure_type == "file":

            mydict ={ "initial_message":"Please enter a number to open the corresponding text file: ",
            }
            correct_response = False
            while correct_response == False:
                try:
                    print ''
                    user_input = int(raw_input(mydict["initial_message"]))
                    print ''
                    return user_input
                    correct_response = True
                except ValueError:
                    print 'Incorrect input. Please enter a number to open the corresponding file. '    

#===================================================================================================================================#

    def user_input_check_directory(CL_list):

        mydict ={ "output1": "You have selected directory ",
                  "output2": " to open."
        }
        
        user_input = initial_user_input("directory") 
            
        correct_input = False

        while correct_input == False:
            if user_input > len(CL_list) or user_input <= 0 or not type(user_input)==int:
                correct_input = False
                print ''
                e = 'Please enter in a number between '+str(1)+' and '+str(len(CL_list))+' to process the corresponding text file,as shown above.'
                print ''
                user_input = initial_user_input("directory") 

            elif 0 < user_input < len(CL_list) and type(user_input) == int:                
                print ''
                print "Correct input."+ mydict["output1"] + str(CL_list[int(user_input)-1]) + mydict["output2"]
                print ''
                correct_input = True
                output = int(user_input)
                return output
                break
            else:
              return False
                

#===================================================================================================================================#

    def user_input_check_file(CL_list):

        mydict ={ "output1": "You have selected file ",
                  "output2": " to open."
        }
        
        user_input = initial_user_input("file") 
            
        correct_input = False

        while correct_input == False:
            if user_input > len(CL_list) or user_input <= 0:
                print ''
                e = 'Please enter in a number between '+str(1)+' and '+str(len(CL_list))+' to process the corresponding text file,as shown above.'
                user_input = int(raw_input(e))
                print ''
            elif 0 < user_input < len(CL_list) and type(user_input) == int:
                print ''
                print mydict["output1"] + str(CL_list[int(user_input)-1]) + mydict["output2"]
                print ''
                correct_input = True
                output = int(user_input)
                return output
            else:
              return False
                        
#===================================================================================================================================#

    def intro():
        root_walk = dir_walk(rootdir)
        all_dirs = root_walk['level_0_dirs']
        print "Welcome to the metabolite python script. Below is a list of directories containing subdirectories and text files.",
        print "\nThe directories are labeled with numbers between 1 and",len(all_dirs),". Please choose a number to open the corresponding directory.",
        print "\nYou will have the option to calibrate the file before filtering the file."
        print ''
        print 'Press ctrl + c to exit the programme at any time.'
        print ''
                        
#===================================================================================================================================#
   
    def file_or_subdirectories_check(user_path_map, counter, current_tree):

      all_paths = os.walk(user_path_map[counter]).next()
      all_dirs = all_paths[1]
      all_files = [n for n in all_paths[2] if '.txt' in n]

      file_names_extension_remove = [remove_extensions(n) for n in current_tree[2]]
      files_to_process  = remove_duplicates(file_names_extension_remove)
      
      if len(all_dirs) != 0:
          user_input = user_input_check_directory(all_dirs)
          
      elif len(all_files) > 1:
          user_input = user_input_check_file(all_files)    

      elif user_input == False:
          return

      return {'user_input':user_input, 'files_to_process':files_to_process, 'all_dirs':all_dirs, 'all_files':all_files}
             
#===================================================================================================================================#
     
    def data_plot(xaxis,yaxis):

        plot_question = "Would you like to plot the processed data?"
        plot_query = query_yes_no(plot_question)

        if plot_query == True:
            plot(xaxis,yaxis)
            print '\nData has been plotted in external window.'
            show()

#===================================================================================================================================#

    def CLI():

        if intro_message == True:
            engage = intro()
       
        end_directory = False
        re_run = False
        
        raise_pca_query = "Would you like to filter/calibrate the data or proceed to PCA? \n\nPlease enter 'y' for PCA or 'n' for filter/calibration"
        pca_query = query_yes_no(raise_pca_query)

        rootdir_output = print_current_dir(rootdir)
        user_path_map = [rootdir] # need to sort this out for a arbitrarily structured directory tree# done. now need to incoperate pca partition.
        user_txt_files = [] # pca partition done, now need to link files with pretty print. user input != cli output

        counter = 0
        
        while counter == True or end_directory == False:

            all_paths = os.walk(user_path_map[counter]).next()
            all_dirs = all_paths[1]
            all_files = all_paths[2]
            
            if len(all_dirs) != 0:
                user_input = user_input_check_directory(all_dirs)
               
            elif len(all_files) !=0:
                user_input = user_input_check_file(all_dirs)   

            elif user_input == False:
                break
            
            dynamical_path = [os.path.join(user_path_map[counter],all_dirs[user_input-1])]#all_dirs[user_input-1])]
            user_path_map.extend(dynamical_path)
            dynamical_output = print_current_dir(dynamical_path[0])
            current_tree = os.walk(dynamical_path[0]).next()

            if 'raw' in all_dirs[user_input-1]:
                engage_pca = False
            elif not 'raw' in all_dirs[user_input-1]:
                engage_pca = True
                
            if not current_tree[1]:
                user_txt_files.extend(current_tree[2])
                end_directory = True

            counter = counter + 1
            
            if end_directory == True: # stick in PCA/filtered loop here - after end dir == True.

                if engage_pca == False:

                    file_names_extension_remove = [remove_extensions(n) for n in current_tree[2]]
                    files_to_process  = remove_duplicates(file_names_extension_remove)
                    
                    process_all_query = "\nWould you like to process all files?\n\nPlease enter 'y' to process all files or 'n' to process individual files"
                    process_all = query_yes_no(process_all_query)

                    cal_query = "Would you like to calibrate before filtering? \nPlease enter yes to calibrate and filter or no to just filter."
                    cal_input = query_yes_no(cal_query)

                    if process_all == True:

                        #=========================================================================#
                              
                        if cal_input == True:

                            output_directory = create_directory(rootdir,all_dirs[user_input-1])
                            ref_data = open_file(dynamical_path[0],files_to_process[0]+"i.txt") #crops/plants etc not raw data?
                           
                            for input_txt_file in files_to_process:
                           
                                if files_to_process[0]+"i.txt"==input_txt_file+"i.txt":

                                    cal_output_data = [ref_data[0]]
                                    print input_txt_file+"i.txt", 'is used as the reference dataset and does not need calibration.'

                                    for m in extension[1:]:

                                        cal_input_data = open_file(dynamical_path[0],input_txt_file+m)
                                        cal = calibrate(ref_data,cal_input_data[0])
                                        cal_output_data.append(cal['calibrated_data'])
                             
                                        print input_txt_file+m, 'calibrated.'
    
                                    fil = filter(cal_output_data) 
                                    save(fil['clean_data'],input_txt_file,output_directory['path'])

                                elif files_to_process[0]+"i.txt"!=input_txt_file+"i.txt":

                                    cal_output_data = []

                                    for m in extension:

                                        cal_input_data = open_file(dynamical_path[0],input_txt_file+m)
                                        cal = calibrate(ref_data,cal_input_data[0])
                                        cal_output_data.append(cal['calibrated_data'])

                                        print input_txt_file+m, 'calibrated.'
                                
                                    fil = filter(cal_output_data)
                                    save(fil['clean_data'],input_txt_file,output_directory['path'])
                                    
                                
                                print 'File ',input_txt_file,' filtered and calibrated.' 

                        #=========================================================================#        
                                
                        elif cal_input == False:

                            output_directory = create_directory(rootdir,all_dirs[user_input-1])

                            for input_txt_file in files_to_process:

                                data_to_fil = []

                                for m in extension:
                                    dataset = open_file(dynamical_path[0],input_txt_file+m)
                                    data_to_fil.append(data_to_fil)
                                    
                                fil = filter(dataset)

                                save(fil['clean_data'],input_txt_file,output_directory['path'])

                                print 'File ',input_txt_file,' filtered.'
                             #   data_plot(fil['mz'],fil['tic']) # do I need this?? would they want images of each filtered file? or just save each plot automatically? (would require a new directory)

                        #=========================================================================#
                                
                    elif process_all == False:

                        user_selected_txt_file = user_input_check_file(files_to_process)
      
                        #=========================================================================#
                                
                        if cal_input == True:
                            
                            ref_data = open_file(dynamical_path[0],files_to_process[0]+"i.txt")
                           
                            if files_to_process[0]+"i.txt"==files_to_process[user_selected_txt_file-1]+"i.txt":

                                cal_output_data = [ref_data[0]]
                                print files_to_process[user_selected_txt_file-1]+"i.txt", 'is used as the reference dataset and does not need calibration.'

                                for m in extension[1:]:

                                    cal_input_data = open_file(dynamical_path[0],files_to_process[user_selected_txt_file-1]+m)
                                    cal = calibrate(ref_data,cal_input_data[0])
                                    cal_output_data.append(cal['calibrated_data'])
                             
                                    print files_to_process[user_selected_txt_file-1]+m, 'calibrated.'

                                fil = filter(cal_output_data) 

                            elif files_to_process[0]+"i.txt"!=files_to_process[user_selected_txt_file-1]+"i.txt":

                                cal_output_data = []

                                for m in extension:

                                    cal_input_data = open_file(dynamical_path[0],files_to_process[user_selected_txt_file-1]+m)
                                    cal = calibrate(ref_data,cal_input_data[0])
                                    cal_output_data.append(cal['calibrated_data'])

                                    print files_to_process[user_selected_txt_file-1]+m, 'calibrated.'
                            
                                fil = filter(cal_output_data)

                            data_plot(fil['mz'],fil['tic'])

                            print 'File ',files_to_process[user_selected_txt_file-1],' filtered and calibrated.' 

                        #=========================================================================#
                                   
                        elif cal_input == False:    

                            dataset = []

                            for m in extension:
                              data = open_file(dynamical_path[0],files_to_process[user_selected_txt_file-1]+m)[0]
                              dataset.append(data)
                     
                            fil = filter(dataset)
                            
                            print "File "+files_to_process[user_selected_txt_file-1]+" been filtered."

                            data_plot(fil['mz'],fil['tic'])

                        #=========================================================================#
                                
                    return {'dynamical_path':dynamical_path,'user_wants_to_filter':cal_input,'choosen_directory':all_dirs[user_input-1]
                    }
                    return cal_input

                elif engage_pca == True: 
                    print ''     
                    print 'Engaging PCA routine...' # don't need to check files - automatically switch to process all mode.
                    print ''
                    all_paths = os.walk(user_path_map[counter]).next()
                    all_dirs = all_paths[1]
                    all_files = all_paths[2]
                   
                    pca_input_data = []

                    for pca_input_file in all_files:
                        datum = open_file(dynamical_path[0],pca_input_file)[0]
                        pca_input_data.append(datum)

                    pca_input_matrix = np.vstack(np.array(pca_input_data))
                   
                    # need 24 x 600 input matrix.
                    binned_pca = binning(pca_input_data)
                   # print np.shape(binned_pca), '<==='
                    #print np.shape(binned_pca) == 24 x 600
                    
                   # pca_engage = princomp(np.array(binned_pca),numpc = 2)
                    pca_engage = princomp(np.array(binned_pca),numpc = 2)
                    variance_value = pca_engage['variance_value']
                   # a = pca_engage['explained_variance']
              #      print variance_value
                  ###  pretty_print(a)
                   # data = pca_engage['projected_data'].T
                   # data_plot([n[0] for n in data],[n[1] for n in data])
          
                    #pretty_print(variance_value)
                    #pca_dataset = open_file(dynamical_path[0], files_to_process[user_input-1])[0]
                    #pca_engage = princomp(pca_dataset, numpc = 2)
                    #print 'Output data:'
                   # print ''
                   # pretty_print(pca_engage['variance_value'])
                   
#===================================================================================================================================#
# END COMMAND LINE INTERPRETER
#===================================================================================================================================#
  
#===================================================================================================================================#
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
# END PLANT METABOLITE SCRIPT
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
#===================================================================================================================================#

#===================================================================================================================================#

    ACTIVATE = CLI()

#===================================================================================================================================#

def Plant_programme_rerun():

    count = 0
    running_metabolite_script = True

    while running_metabolite_script == True:

        if count == 0:
            intro_message = True
        else:
            intro_message = False
            
        Plant_metabolite_python_script()

        running_script_query = "\nWould you like to process further files? Please enter yes or no."
        running_script_query_input = query_yes_no(running_script_query)

        if running_script_query_input == False:

            running_metabolite_script == False
            print ''
            print 'Bye bye, have a nice day.'
            break
          
        elif running_script_query_input == True:
            print ''
            print 'Program will re-run.'
            print ''
            print '================================ RESTART ================================'
            print ''

        count = count + 1

#===================================================================================================================================#

def main():
 
    ENGAGE = Plant_programme_rerun()

if __name__ == "__main__":
   try:
      main()
   except KeyboardInterrupt:
      print ''
      print 'Bye bye, have a nice day.'
      pass

#===================================================================================================================================#
