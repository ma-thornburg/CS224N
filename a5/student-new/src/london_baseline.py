# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.
import utils as utils 

total, correct = utils.evaluate_places('birth_dev.tsv', ['London' for x in range(0, 500)])

print('Result if just predicted London: correct:{correct}, total:{total}: percentage:{percentage}'.format(total=total, correct=correct, percentage=correct/total))