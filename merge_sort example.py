counter = 0
def merge_sort(A):
	if len(A) < 2:
		return A
	left = A[0:len(A)/2]
	right = A[len(A)/2:len(A)]
	left = merge_sort(left)
	right = merge_sort(right)
	return merge(left, right)

def merge(left, right):
	result = []
	global counter
	while len(left) > 0 and len(right) > 0:
		if left[0] <= right[0]:
			result.append(left[0])
			del left[0]
		else:
			counter += len(left)
			result.append(right[0])
			del right[0]
	if len(left) > 0:
		result.extend(left)
	if len(right) > 0:
		result.extend(right)
	return result

M = [5, 8, 7, 2, 1, 4, 3, 10, 6, 9]
#M = [4, 1, 2, 3, 5, 6, 7, 8, 9, 10]
#M = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
A = merge_sort(M)
print A