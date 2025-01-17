from typing import List

class Solution:

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

    matrix = [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    matrix2 = [[1,2,3], [4,5,6], [7,8,9]]
    matrix3 = [[5,1,9,11], [2,4,8,10], [13,3,6,7], [15,14,12,16]]
    nums = [1,2,3,4]
    nums2 = [-1,1,0,-3,3]
    nums3 = [1,2,3,1]
    nums4 = [1,1,1,3,3,4,3,2,4,2]
    matrix4 = [[-1,2,-3,4]]
    #strs = ["flower","flow","flight"]
    #nums = [0,1,2,4,5,7]
    #nums2 = [0,2,3,4,6,8,9]

    jewels = "aA"
    stones = "aAAbbbb"

    jewels2 = "z"
    stones2 = "ZZ"

    ransomNote = "a"
    magazine = "b"
    ransomNote2 = "aa"
    magazine2 = "ab"
    ransomNote3 = "aa"
    magazine3 = "aab"

    s = "anagram"
    t = "nagaram"

    s2 = "rat"
    t2 = "car"

    intervals1 = [[1,3], [2,6], [8,10], [15,18]]
    intervals2 = [[1,4], [4,5]]
    intervals3 = [[1,4], [0,4]]
    intervals4 = [[1,3]]

    text = "nlaebolko"
    text2 = "loonbalxballpoon"
    text3 = "leetcode"
    text4 = "1"

    tsnums = [2,7,11,15]
    target = 9

    board =[["5", "3", ".", ".", "7", ".", ".", ".", "."]
        , ["6", ".", ".", "1", "9", "5", ".", ".", "."]
        , [".", "9", "8", ".", ".", ".", ".", "6", "."]
        , ["8", ".", ".", ".", "6", ".", ".", ".", "3"]
        , ["4", ".", ".", "8", ".", "3", ".", ".", "1"]
        , ["7", ".", ".", ".", "2", ".", ".", ".", "6"]
        , [".", "6", ".", ".", ".", ".", "2", "8", "."]
        , [".", ".", ".", "4", "1", "9", ".", ".", "5"]
        , [".", ".", ".", ".", "8", ".", ".", "7", "9"]]

    print(Solution().isValidSudoku(board))

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
