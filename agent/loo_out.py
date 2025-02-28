import os

from agent.state import OverallState


def calculate_loo_accuracy(results):
    test_trues = [d['test_true'] for d in results]
    test_preds = [d['test_pred'] for d in results]
    correct_predictions = sum(t == p for t, p in zip(test_trues, test_preds))
    accuracy = correct_predictions / len(test_trues)
    return accuracy

def loo_out(state:OverallState):
    output_folder = state['output_folder']
    all_results = state['loo_log']
    test_accuracy = calculate_loo_accuracy(all_results)
    current_output = f'Final Test Accuracy: {test_accuracy}' +'\n'
    with open(os.path.join(output_folder,'loo_out_log.txt'),'a') as f:
        f.write(current_output)
    return {'test_accuracy':test_accuracy}