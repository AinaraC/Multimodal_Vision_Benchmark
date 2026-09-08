import time
import csv
import subprocess
import jtop
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Orin benchmark a model')
    parser.add_argument('--filepath', help='file name to save metrics')
    parser.add_argument('--wait',default=0, help='wait seconds aprox')
    args = parser.parse_args()
    return args

class JetsonMetrics():
    def __init__(self, file_path, wait_sec):
        self.wait_sec = wait_sec
        self.file_path = file_path

    def get_jetson_metrics_jtop(self):
        with jtop.jtop() as jetson:
            if jetson.ok():
                data = jetson.stats
                metrics = {
                    'gpu_usage': data['GPU'],
                    'cpu_1': data['CPU1'],
                    'cpu_2': data['CPU2'],
                    'cpu_3': data['CPU3'],
                    'cpu_4': data['CPU4'],
                    'cpu_5': data['CPU5'],
                    'cpu_6': data['CPU6'],
                    'cpu_7': data['CPU7'],
                    'cpu_8': data['CPU8'],
                    'gpu_temp': data['Temp gpu'],
                    'cpu_temp': data['Temp cpu'],
                    'tj_temp': data['Temp tj'],
                    'power_tot': data['Power TOT'],
                    'power_model': data['Power VDD_CPU_GPU_CV'],
                    'RAM': data['RAM'],   
                    'elapsed_time': 0
                }
                return metrics
                
    def write_metrics_to_csv_jtop(self):
        fieldnames = ['gpu_usage','cpu_1','cpu_2','cpu_3','cpu_4','cpu_5','cpu_6','cpu_7','cpu_8', 'gpu_temp', 'cpu_temp','tj_temp', 'power_tot','power_model', 'RAM', 'elapsed_time']
        df = pd.DataFrame(columns=fieldnames)
        start_time = time.time()
        while True:
            metrics = self.get_jetson_metrics_jtop()
            metrics['elapsed_time'] = time.time() - start_time
            df = df.append(metrics, ignore_index=True)
            df.to_csv(self.file_path)
            time.sleep(self.wait_sec)   

if __name__ == '__main__':
    args = parse_args()
    jetson_metrics = JetsonMetrics(args.filepath, args.wait)
    jetson_metrics.write_metrics_to_csv_jtop() 

