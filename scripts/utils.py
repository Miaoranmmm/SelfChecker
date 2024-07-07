import json
import os

def write_json(new_id, new_data, filename):
    if not os.path.exists(filename):
        json_object = json.dumps({new_id: new_data}, indent=2)
        with open(filename, 'w') as outfile:
            outfile.write(json_object)
    else:
        with open(filename,'r+') as outfile:
            # First we load existing data into a dict.
            file_data = json.load(outfile)
            # if new_id in file_data:
            #     del file_data[new_id]
                
            # Join new_data with file_data inside emp_details
            file_data[new_id] = new_data
            # Sets file's current position at offset.
            outfile.seek(0)
            # convert back to json.
            json.dump(file_data, outfile, indent = 2)