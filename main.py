from collections import Counter, deque
from typing import List

class Solution:

    def isValidParentheses(self, s: str) -> bool:
        """
        Given a string s containing just the character '(', ')', '{', '}', '[' and ']',
        determine if the input string is valid.

        an input string is valid if:
        1. open brackets must be closed by the same type of brackets.
        2. open brackets must be closed in the correct order.
        3. every close bracket has a corresponding open bracket of the same type.

        Input: s = "()"
        Output: True

        Input: s = "(]"
        Output: False
        """

        if not s:
            return False

        if len(s) % 2 != 0:
            return False

        stack = []
        bracket_pairs = {
            ')': '(',
            '}': '{',
            ']': '['
        }

        for char in s:
            if char in bracket_pairs.values():  # If it's an opening bracket
                stack.append(char)
            elif char in bracket_pairs.keys():  # If it's a closing bracket
                if not stack or stack.pop() != bracket_pairs[char]:
                    return False
            else:
                # If the character is not a valid bracket
                return False

        # If the stack is empty, the brackets were matched correctly
        return not stack

    def calPoints(self, operations: List[str]) -> int:
        """
        You are keeping the scores for a baseball game with strange rules.
        At the beginning of the game, you start with an empty record. You are given a list of strings operations,
        where operations[i] is the i-th operation you must apply to the record and is one of the following:

        integer x: record a new score of x
        '+': record a new score that is the sum of the previous two scores
        'D': record a new score that is the double of the previous score
        'C': invalidate the previous score, removing it from the record

        return the sum of all the scores on the record after applying all the operations.

        the test cases are generated such that the answer and all intermediate calc fit in a 32-bit integer.
        all operations are valid.

        Input: ops = ["5", "2", "C", "D", "+"]
        Output: 30
        """
        record = [0] * len(operations)
        ops = deque(operations)

        print(ops)
        while ops:
            op = ops.popleft()

            if op == "C":
                print("Remove last operation: \n", record)
                record.pop()

            elif op == "+":
                print(f"Sum {record[-2]} and {record[-1]}\n", record)
                record.append(int(record[-2]) + int(record[-1]))

            elif op == "D":
                print(f"Double {record[-1]}\n", record)
                record.append(int(record[-1]) * 2)

            else:
                print(f"Add {op}\n", record)
                record.append(int(op))

        return sum(record)

    def trap(self, height: List[int]) -> int:
        """
        Given n non-negative integers representing an elevation map where the width of each bar is 1
        compute how much water it can trap after raining.

        Input: height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
        Output: 6
        """

        left, right = 0, len(height) - 1
        max_l, max_r = height[left], height[right]
        area = 0

        while left < right:
            if max_l <= max_r:
                # always increment left and update max_l
                left += 1

                if height[left] < max_l:
                    # water is trapped here
                    area += max_l - height[left]
                else:
                    # update max_l if current height is greater
                    max_l = height[left]

            else:
                # always decrement right and update max_r
                right -= 1

                if height[right] < max_r:
                    # water is trapped here
                    area += max_r - height[right]
                else:
                    # update max_r if current height is greater
                    max_r = height[right]

            # debug print
            print(f"left={left}, right={right}, max_l={max_l}, max_r={max_r}, area={area}")

        return area

    def maxArea(self, height: List[int]) -> int:
        """
        Q. 11: Container with Most Water.
        You are given an integer array height of length n.
        There are n vertical lines drawn such that the two endpoints of the i-th line are (i,0) and (i, height[i])

        Find the two lines that together with the x-axis form a container, such that the continer contains
        the most water.

        Return the maximum amount of water a container can store.

        Notice that you may not slant the container.

        Input: height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
        Output = 49
        """

        n = len(height)
        area = 0        # determined by max right that is <= max left * index + 1
        left, right = 0, n - 1
        max_l, max_r = height[left], height[right]

        while left < right:

            area = max(area, min(max_l, max_r) * (right - left))

            if max_l < max_r:
                left += 1
                max_l = max(max_l, height[left])
            else:
                right -= 1
                max_r = max(max_r, height[right])

        return area

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """
        Q: 15. 3Sum
        Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that
        i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

        Notice that the solution set must not contain duplicate triplets.

        Input: nums = [-1, 0, 1, 2, -1, -4]
        Output = [[-1, -1, 2], [-1, 0, 1]]
        Explanation: nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
                    nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
                    nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.

        The distinct triplets are [-1,0,1] and [-1,-1,2].
        Notice that the order of the output and the order of the triplets does not matter.
        """
        nums.sort()
        output = []

        for fixed_num in range(len(nums) - 2):
            if fixed_num > 0 and nums[fixed_num] == nums[fixed_num - 1]:
                continue

            left, right = fixed_num + 1, len(nums) - 1

            while left < right:
                three_sum = nums[fixed_num] + nums[left] + nums[right]

                if three_sum == 0:
                    output.append([nums[fixed_num], nums[left], nums[right]])

                    left += 1
                    right -= 1

                    while left < right and nums[left] == nums[left - 1]:
                        left += 1
                    while left < right and nums[right] == nums[right + 1]:
                        right -= 1

                elif three_sum < 0:
                    left += 1
                else:
                    right -= 1

        return output

    def isPalindrome(self, s: str) -> bool:
        """
        Q: 125. Valid Palindrome
        A phrase is a palindrome if it reads the same forwards and backwards.
        We're lowercasing all chars, removing all non characters.

        Given string s, return true if its a palindrome, false otherwise.

        Input: s = "A man, a plan, a canal: Panama"
        Output: true
        Explanation: "amanaplanacanalpanama" is a palindrome
        """

        left, right = 0, len(s)-1

        while left < right:
            while left < right and not s[left].isalnum():
                left += 1
            while left < right and not s[right].isalnum():
                right -= 1

            if s[left].lower() != s[right].lower():
                return False

            left += 1
            right -= 1

        return True

    def twoSum2(self, numbers: List[int], target: int) -> List[int]:
        """
        Q: 167. Two Sum II - Input Array Is Sorted
        Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, find two numbers
        such that they add up to a specific target number.

        Let these two numbers be numbers[index1] and numbers[index2] where 1 <= index1 < index2 <= numbers.length

        Return the indices of the two numbers, index1 and index2, added by one as an integers array [index1, index2]
        of length 2.

        The tests are generated such that there is exactly one solution. You may not use the same element twice.

        Your solution must use only constant extra space.

        Input: numbers = [2, 7, 11, 15], target = 9
        Output: [1, 2]
        Explanation: The sum of 2 and 7 is 9. therefore, index1 = 1, index2 = 2. return [1, 2]
        """
        seen = {}

        for i in range(len(numbers)):
            test_num = numbers[i]
            complement = target - test_num

            if complement in seen:
                return [seen[complement] + 1, i + 1]
            seen[test_num] = i

        return []

    def reverseString(self, s: List[str]) -> None:
        """
        Q: 344. Reverse String
        Write a function that reverses a string. The input string is given as na array of characters .

        You must do this by modifying the input array in place with O(1) extra memory.

        Input: s = ["h","e","l","l","o"]
        Output: ["o","l","l","e","h"]
        """

        print(s)

        for i in range(len(s)//2):
            s[i], s[len(s)-i-1] = s[len(s)-i-1], s[i]

        print(s)

    def sortedSquares(self, nums: List[int]) -> List[int]:
        """
        Q: 977. Squares of a Sorted Array
        Given an integer array nums sorted in non-decreasing order, return an array of the
        squares of each number sorted in non-decreasing order.

        Input: nums = [-4, -1, 0, 3, 10]
        Output = [0, 1, 9, 16, 100]
        """
        output = []

        for num in nums:
            output.append(num ** 2)

        return sorted(output)

    def longestConsecutive(self, nums: List[int]) -> int:
        """
        Q.128 Longest Consecutive Sequence

        Given an unsorted array of integers, return the length of the longest consecutive elements sequence.

        You must write an algorithm with O(n) runtime complexity.

        nums = [100, 4, 200, 1, 3, 2]
        output: 4


        """

        # we have a running update of what is in sequence and how long it is
        # if our current number + or - 1 is in the number set, then it's in the sequence.
        # if we have no sequence at the end, the number is 1 unless nums is empty
        # nums = [9,1,4,7,3,-1,0,5,8,-1,6]
        # for above nums, -1 0 1 are in sequence but are broken by a missing 2 - still considered by the code to be in sequence

        '''
            nums_set = set(nums)
            seen_nums = set()
            s_count = 0
    
            if not nums_set:
                return s_count
    
            for i in range(len(nums)):
                num = nums[i]
                np1 = nums[i] + 1
                nm1 = nums[i] - 1
    
                print(f'Test {i}: checking index {i}, num {nums[i]} \n')
    
                if (np1 in nums_set or nm1 in nums_set) and num not in seen_nums:
                    s_count += 1
                    seen_nums.add(num)
                    print(f'Test {i}: {np1} or {nm1} found in {nums_set}. s_count now {s_count} \n')
                else:
                    print(f'Test {i}: {np1} or {nm1} not found in {nums_set}. \n')
    
            # if we've seen all numbers and still have no sequence, it is its own sequence
            if s_count < 1:
                s_count += 1
    
            return s_count
        '''

        if not nums:
            return 0

        nums_set = set(nums)
        lstreak = 0

        for num in nums_set:

            if num - 1 not in nums_set:
                cnum = num
                cstreak = 1

                while cnum + 1 in nums_set:
                    cnum += 1
                    cstreak += 1

                lstreak = max(lstreak, cstreak)

        return lstreak

    def majorityElement(self, nums: List[int]) -> int:
        """
        Q. 169 Majority Element
        Given an array nums of size n, return the majority element.

        The majority element is the element that appears more than n/2 times.
        You may assume that the majority element always exists in the array.
        """

        majority = Counter(nums)

        print(majority)

        return max(majority.keys(), key=majority.get)

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        """
        Q: 49. Group Anagrams

        Given an array of strings strs, group the anagrams together.
        You can return the answer in any order.

        Input: strs = ["eat","tea","tan","ate","nat","bat"]
        Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

        Explanation:
        There is no string in strs that can be rearranged to form "bat".
        The strings "nat" and "tan" are anagrams as they can be rearranged to form each other.
        The strings "ate", "eat", and "tea" are anagrams as they can be rearranged to form each other.
        """

        # we count the letters that are in each word
        # check if the next word has the same amount
        # if they do they are an anagram

        anagrams = {}

        for word in strs:

            sorted_word = "".join(sorted(word))

            if sorted_word not in anagrams:
                anagrams[sorted_word] = []
            anagrams[sorted_word].append(word)

        return list(anagrams.values())

    def isValidSudoku(self, board: List[List[str]]) -> bool:
        """
        Q: 36. Valid Sudoku

        Determine if a 9x9 sudoku board is valid. Only the filled cells need to be validated according to;
        1. each row must contain digits 1-9 without repetition
        2. each column likewise
        3. each of the nine 3x3 sub-boxes of the grid must contain digits 1-9 without repetition.
        """

        """
        a much nicer solution that doesnt have unnecessary repetition:
        the key here is finding a way to number the boxes and properly checking numbers across rows / columns
        
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        box = [set() for _ in range(9)]
        
        for i in range(9)
            for j in range(9)
                num = board[i][j]
                
                if num == '.':
                    continue
                    
                box_index = (i // 3) * 3 + (j // 3)
                
                if num in rows[i] or num in cols[j] or num in box[box_index]:
                    return False
                
                rows[i].add(num)
                cols[j].add(num)
                box[box_index].add(num)
            
        return True
        """
        # determine if across is valid
        # determine if up-down is valid

        # check sudoku
        for i in range(len(board)):

            # reset both sets once we've iterated over a line
            h_line = set()
            v_line = set()

            for j in range(len(board[0])):

                # if board has a number on this line
                if board[i][j] != '.':

                    # check horizontal line
                    if board[i][j] in h_line:
                        return False
                    else:
                        h_line.add(board[i][j])

                # if board has number on this line
                if board[j][i] != '.':

                    # check vertical line
                    if board[j][i] in v_line:
                        return False
                    else:
                        v_line.add(board[j][i])

        for row in range(0, 9, 3):
            for col in range(0, 9, 3):
                sub_box = set()

                for i in range(row, row+3):
                    for j in range(col, col+3):
                        if board[i][j] != ".":
                            if board[i][j] in sub_box:
                                return False
                            else:
                                sub_box.add(board[i][j])

        return True

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """
        Q: 1. Two Sum

        Given an array of integers nums and an integer target, return indices of the two numbers
        such that they add up to target.

        You may assume that each input would have exactly one solution, and you may not use the same
        element twice.

        You can return the answer in any order.
        """

        for i in range(len(nums)):                          # look through the array
            test_num = nums[i]                              # grab a test number

            complement = target - test_num
            if complement in nums[i+1:]:
                return [i, nums.index(complement, i+1)]   # return our pair

        return []

    def maxNumberofBalloons(self, text: str) -> int:
        """
        Q: 1189. Maximum number of balloons

        Given a string Text, you want to use the characters of Text to form as many instances of the word
        "balloon" as possible.

        You can use each character in Text at most once. Return the maximum number of instances.
        """
        bank = {}
        balloons = 0

        # if we can't have one balloon why bother partying
        if len(text) < 7:
            return 0

        # populate bank
        for char in text:
            if char in bank:
                bank[char] += 1
            else:
                bank[char] = 1

        # b a l l o o n
        while True:
            for char in "balloon":
                if char in bank and bank[char] > 0:
                    bank[char] -= 1
                else:
                    return balloons
            balloons += 1

    def isAnagram(self, s: str, t: str) -> bool:
        '''
        Q: 242. Valid Anagram

        Given two strings s and t, return True if t is an anagram of s, and false otherwise.
        '''

        count_s = {}
        count_t = {}

        for char in s:
            if char in count_s:
                count_s[char] += 1
            else:
                count_s[char] = 1

        for char in t:
            if char in count_t:
                count_t[char] += 1
            else:
                count_t[char] = 1

        return count_s == count_t

    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        '''
        Q: 383. Ransom Note
        Given two strings ransomNote and magazine, return True if ransomNote can be constructed by using
        the letters from magazine, else return False.

        Each letter in magazine can only be used once in ransomNote.
        '''

        '''
        improved answer:
        for letter in set(ransomNote):
            if ransomNote.count(letter) > magazine.count(letter):
                return False
        return True
        '''
        bank = {}

        # go through the magazine and find out how many letters we have to play with
        for letter in magazine:
            if letter not in bank:
                bank[letter] = 1
            elif letter in bank:
                bank[letter] += 1

        # use the bank of letters to construct the note
        for letter in ransomNote:
            if letter in bank and bank[letter] > 0:
                bank[letter] -= 1
            else:
                return False

        return True

    def containsDuplicate(self, nums: List[int]) -> bool:
        '''
        Q: 217. Contains Duplicate

        Given an integer array nums, return true if any value appears at least twice in the array.
        Return false if every element in the array is distinct
        '''

        seen_numbers = set()

        for num in nums:
            if num in seen_numbers:
                return True
            else:
                seen_numbers.add(num)

        return False

    def numJewelsInStones(self, jewels: str, stones: str) -> int:
        '''
        Q: 771. Jewels and Stones
        You're given strings Jewels representing the types of stones that are jewels
        and Stones representing the stones you have. Each character in stones is a type of stone
        you have. You want to know how many of the stones you have are also jewels.

        Letters are case-sensitive, so "a" is different from "A"
        '''

        hoard = 0
        for item in stones:
            if item in jewels:
                hoard += 1
            else:
                continue
        return hoard

    def rotate(self, matrix: List[List[int]]) -> None:
        """
        You are given an N x N 2D matrix representing an image, rotate the image by 90 degrees (clockwise).

        You have to rotate the image in-place, which means you have to modify the input 2D matrix directly.
        DO NOT allocate another 2D matrix and do the rotation

        Do not return anything, modify matrix in-place instead.

        Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
        Output: [[7,4,1],[8,5,2],[9,6,3]]
        """

        n = len(matrix)

        for i in range(n):
            for j in range(i + 1, n):  # Only swap elements above the diagonal
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

        for i in range(n):
            matrix[i].reverse()

        return matrix

    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        rows = len(matrix)
        cols = len(matrix[0])

        right = cols - 1
        left = 0
        top = 0
        bottom = rows - 1

        output = []

        while len(output) < rows * cols:

            for col in range(left, right + 1):      # initially 0->cols
                output.append(matrix[top][col])     # append
            top += 1                                # shrink the top boundary

            for row in range(top, bottom + 1):      # initially 0->rows
                output.append(matrix[row][right])   # append
            right -= 1                              # shrink the right boundary

            if len(output) >= rows * cols:          # check if its single row
                break

            for col in range(right, left - 1, -1):  # initially cols->-1
                output.append(matrix[bottom][col])
            bottom -= 1

            for row in range(bottom, top - 1, -1):  # initially rows->-1
                output.append(matrix[row][left])
            left += 1

        return output

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:

        intervals.sort(key=lambda x: x[0])

        merged_intervals = []
        temp_interval = intervals[0]  # Start with the first interval

        for i in range(1, len(intervals)):
            # If the current interval overlaps or touches the temp_interval, merge them
            if intervals[i][0] <= temp_interval[1]:
                temp_interval = [temp_interval[0], max(temp_interval[1], intervals[i][1])]
            else:
                # No overlap, add temp_interval to the result and move to the next interval
                merged_intervals.append(temp_interval)
                temp_interval = intervals[i]

        # Don't forget to append the last interval
        merged_intervals.append(temp_interval)

        return merged_intervals

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        """
        Given an integer array nums, return an array answer such that answer[i] is equal to the product of
        all the elements of nums[i] except nums[i].

        The product of any prefix of nums is guaranteed to fit in a 32-bit integer.

        You must write an algorithm that runs in 0(n) time and without using the division operation.

        Input: nums = [1,2,3,4]
        Output: [24,12,8,6]

        Input: nums = [-1,1,0,-3,3]
        Output: [0,0,9,0,0]

        for nums[1] first would be 2x3x4 = 24
        nums[2] 1x3x4 = 12 etc

        a loop which will iterate over the string but skip itself
        each time it goes across

        :param nums:
        :return:
        """

        '''
            product = 1
            test_int = 0
            output = []
            counter = 0
    
            while test_int < len(nums):                     # iterate over the nums list to test each int
                for counter in range(len(nums)):            # iterate over nums to multiply each other int
                    if nums[test_int] != nums[counter]:     # if the current counter isn't the tested integer
                        product *= nums[counter]            # product them
                output.append(product)                      # once done, set the output for that test_int as the product
                product = 1                                 # reset product
                test_int += 1                               # go next
    
            return output
        '''

        # prefix and suffix product
        '''
            test_int = 0
            output = []
    
            while test_int < len(nums):     # iterate over each test integer
                if test_int == 0:                           # if the test int is the first element
                    product = math.prod(nums[test_int:])
                elif test_int == len(nums):                 # if the test int is the last element
                    product = math.prod(nums[:-test_int])
                else:                                       # all but the test element
                    product = math.prod(nums[test_int+1:]) * math.prod(nums[:test_int])
                output.append(product)
                test_int += 1
    
            return output
        '''

        n = len(nums)
        output = [1] * n

        prefix = 1
        for i in range(n):
            output[i] = prefix
            prefix *= nums[i]

        suffix = 1
        for i in range(n -1, -1, -1):
            output[i] *= suffix
            suffix *= nums[i]

        return output

    def summaryRanges(self, nums: List[int]) -> List[str]:
        # YOU ARE GIVEN A SORTED UNIQUE INTEGER ARRAY nums
        # A RANGE [a,b] IS THE SET OF ALL INTEGERS FROM a TO b INCLUSIVE
        # RETURN THE SMALLEST SORTED LIST OF RANGES THAT COVER ALL THE NUMBER IN THE ARRAY EXACTLY.
        # EACH ELEMENT OF nums IS COVERED BY EXACTLY ONE OF THE RANGES
        # AND THERE IS NO INTEGER x THAT IS IN ONE OF THE RANGES BUT NOT IN nums

        """
        input: nums = [0,1,2,4,5,7]
        output: ["0->2", "4->5", "7"]
        explanation: the ranges are 0->2, 4->5 and 7

        you want to check if numbers are increasing by 1 each time
        if the number is increasing by 1, then go next until it doesn't or your out of bounds
        """

        i = 0
        output = []
        first_num = 0

        if not nums:                                                # if nums is empty, return an empty list
            return output

        while i < (len(nums)):                                      # loop until we've gone through the entire list
            if (i+1 < len(nums)) and (nums[i+1] == nums[i]+1):      # if in bounds and the next number is exactly 1 higher than current
                i += 1                                              # increment counter and continue
            elif nums[first_num] == nums[i]:                        # check if first_num and i are the same
                output.append(str(nums[i]))
                i += 1
                first_num = i
            else:                                                   # if it isn't exactly 1 higher append to output (start -> i)
                output.append(str(nums[first_num]) + "->" + str(nums[i]))
                i += 1
                first_num = i

        return output

    def longestCommonPrefix(self, strs: List[str]) -> str:

        '''
        if not strs:                                        # check if its empty first and immediately exit
            return ""

        prefix = strs[0]                                    # prefix starts as the first character

        for string in strs[1:]:                             # compare each string in the list
            while not string.startswith(prefix):            # trim the prefix until it matches the start of string
                prefix = prefix[:-1]                        # remove last character from prefix
                if not prefix:
                    return ""

        return prefix
        '''

        if not strs:                                # if strs is empty, return ""
            return ""

        for i in range(len(strs[0])):               # iterate over the first word in the list
            letter = strs[0][i]                     # get the letter to compare

            for word in strs[1:]:                   # iterate over the other words in the list
                if i >= len(word) or word[i] != letter:     # if we look out of bounds or the letter isn't in word
                    return strs[0][:i]              # return what we have so far

        return strs[0]                              # otherwise return ""


if __name__ == '__main__':

    print("\nTest 1:")
    print(Solution().isValidParentheses(s = "()"))

    print("\nTest 2:")
    print(Solution().isValidParentheses(s = "()[]{}"))

    print("\nTest 3:")
    print(Solution().isValidParentheses(s = "(]"))

    print("\nTest 4:")
    print(Solution().isValidParentheses(s = "([])"))

    print("\nTest 5: Unmatched Opening / Closing")
    print(Solution().isValidParentheses(s = "[(])"))

    print("\nTest 5: Unmatched Opening / Closing")
    print(Solution().isValidParentheses(s="([{}]]])"))

    """    print("\nTest 1:")
    print(Solution().calPoints(["5","2","C","D","+"]))

    print("\nTest 2:")
    print(Solution().calPoints(["5","-2","4","C","D","9","+","+"]))

    print("\nTest 3:")
    print(Solution().calPoints(["1","C"]))"""

    """    print("\nTest Case 1")
    print(Solution().trap(height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))

    print("\nTest Case 2")
    print(Solution().trap(height = [4, 2, 0, 3, 2, 5]))

    print("\nTest Case 3")
    print(Solution().trap(height = [0, 0, 0, 0]))

    print("\nTest Case 4")
    print(Solution().trap(height = [1]))"""

    """    print("\nTest Case 1")
    print(Solution().maxArea(height = [1, 8, 6, 2, 5, 4, 8, 3, 7]))

    print("\nTest Case 2")
    print(Solution().maxArea(height = [1, 1]))

    print("\nTest Case 3")
    print(Solution().maxArea(height = [10, 8, 6, 2, 5, 4, 8, 3, 0]))"""

    """    print("\nTest Case 1")
    print(Solution().threeSum(nums = [-1, 0, 1, 2, -1, -4]))

    print("\nTest Case 2")
    print(Solution().threeSum(nums = [0, 1, 1]))

    print("\nTest Case 3")
    print(Solution().threeSum(nums = [0, 0, 0]))

    print("\nTest Case 4")
    print(Solution().threeSum(nums = [1, -1, 0, 1]))"""

    """    print("\nTest Case 1")
    print(Solution().isPalindrome(s = "A man, a plan, a canal: Panama"))

    print("\nTest Case 2")
    print(Solution().isPalindrome(s = "race a car"))

    print("\nTest Case 2")
    print(Solution().isPalindrome(s = " "))"""

    """
    print("Test Case 1\n")
    print(Solution().twoSum2(numbers = [2, 7, 11, 15], target = 9))

    print("Test Case 2\n")
    print(Solution().twoSum2(numbers = [2, 3, 4], target = 6))

    print("Test Case 3\n")
    print(Solution().twoSum2(numbers = [-1, 0], target = -1))
    """

    #print(Solution().reverseString(string3))

    #print(Solution().sortedSquares(ssnum))

    #print(Solution().longestConsecutive(numbers4))

    #print(Solution().majorityElement(majelement3))

    #print(Solution().groupAnagrams(strs2))

    #print(Solution().isValidSudoku(board))

    #print(Solution().twoSum(tsnums, target))

    #print(Solution().maxNumberofBalloons(text4))

    #print(Solution.isAnagram(s))

    #print(Solution().canConstruct(ransomNote3, magazine3))

    #print(Solution().containsDuplicate(nums3))

    #print(Solution().numJewelsInStones(jewels, stones))

    #print(Solution().rotate(matrix4))

    #print(Solution().spiralOrder(matrix2))

    #print(Solution().merge(intervals2))

    # print(Solution().productExceptSelf(nums2))
    # print(math.prod(nums[test_int:]) * math.prod(nums[:test_int]))

    # print(Solution().summaryRanges(nums))

    # print(Solution().longestCommonPrefix(strs))
