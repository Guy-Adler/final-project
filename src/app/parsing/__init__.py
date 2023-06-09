from .prediction_mapping import MAPPINGS
from src.config import CHARACTERS_AMOUNT


# region Custom Errors


class InvalidMrzLineLengthException(Exception):
    """
    A custom exception to be raised when the MRZ is too short or too long.
    """

    def __init__(self):
        self.message = "MRZ line must be 44 characters long"
        super().__init__(self.message)


class NotACharException(Exception):
    """
    A custom exception to be raised when a string does not have exactly one character.
    """

    def __init__(self):
        self.message = "String must be a character"
        super().__init__(self.message)


# endregion

ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUMERIC = "0123456789"
ARROW = '<'


class Parser:
    """
    A parser for an MRZ string.
    Based on the specification on
    https://www.icao.int/publications/Documents/9303_p4_cons_en.pdf, section 4.2.2
    """

    def __init__(self, mrz: str):
        # Initialize parser variables.
        self._data = {  # Will hold the passport data.
            'issuer_iso': None,  # ISO country code of issuing country of passport; str
            'name': None,  # First and last name of passport holder; {first: str, last: str}
            'passport_number': None,  # The passport number; str
            'is_passport_number_valid': None,  # Whether the check digit for the passport number matches; bool
            'nationality_iso': None,  # ISO country code of passport holder nationality; str
            'date_of_birth': None,  # Date of birth of passport holder; {day: str, month: str, year: str}
            'is_date_of_birth_valid': None,  # Whether the check digit for the date of birth is valid; bool
            'sex': None,  # Sex of passport holder; str | None
            'date_of_expiry': None,  # Date of expiry of passport;  {day: str, month: str, year: str}
            'is_date_of_expiry_valid': None,  # Whether the check digit for the date of expiry is valid; bool
            'personal_number': None,  # Personal number of passport holder (optional); str
            'is_personal_number_valid': None,  # Whether the check digit for the personal number is valid; bool
            'is_mrz_valid': None  # Whether the check digit for the entire MRZ is valid; bool
        }
        self._meta = {  # Information about the parsing
            # Is there reason to think the name was not parsed correctly
            'should_check_name': False,
            # Is there reason to think the passport number was not parsed correctly
            'should_check_passport_number': False,
            # Is there reason to think the date of birth was not parsed correctly
            'should_check_date_of_birth': False,
            # Is there reason to think the date of expiry was not parsed correctly
            'should_check_date_of_expiry': False,
            # Is there reason to think the personal number was not parsed correctly
            'should_check_personal_number': False,
            # Is there reason to think the mrz in general was not parsed correctly
            'should_check_mrz': False
        }
        # Save the MRZ lines as instance variables
        if len(mrz) == CHARACTERS_AMOUNT * 2:  # 44 characters per line, 88 total
            self._mrz_line_1: str = self.validate_length(mrz[:CHARACTERS_AMOUNT])
            self._mrz_line_2: str = self.validate_length(mrz[CHARACTERS_AMOUNT:])
        else:
            raise InvalidMrzLineLengthException()

        self._preprocessing()  # Preprocess the lines to correct invalid predictions and make the parsing more accurate
        self._parse()  # Start the parsing process

    def get_data(self):
        return self._data

    def get_meta(self):
        return self._meta

    @staticmethod
    def validate_length(mrz_line: str) -> str:
        """
        Make sure the length of the mrz_line is exactly 44 characters.
        :param mrz_line: An MRZ line to validate.
        """
        if len(mrz_line) == CHARACTERS_AMOUNT:
            return mrz_line
        raise InvalidMrzLineLengthException()

    @staticmethod
    def to_int(char: str) -> int:
        """
        Convert a character to an integer, for the check digit calculation.
        :param char: A character (length 1 string)
        """
        if len(char) != 1:
            raise NotACharException()
        if char in NUMERIC:
            return int(char)  # map '0' -> 0, ..., '9' -> 9
        elif char in ALPHA:
            return ord(char) - ord('A') + 10  # Use ASCII character codes to map A -> 10, ..., Z -> 35
        else:
            # Should only occur when char == '<'.
            return 0

    @staticmethod
    def check_digit(digit: str, number: str) -> bool:
        """
        Calculate the check digit on a string,
        as shown in https://www.icao.int/publications/Documents/9303_p3_cons_en.pdf, section 4.9
        :param digit: The check digit
        :param number: The string to check against
        """
        if digit not in NUMERIC:
            # Check digit is invalid
            return False
        check_sum = 0  # Will keep the weighted sum of the digits
        pattern = [7, 3, 1]  # The repeating weight pattern
        for idx, char in enumerate(number):
            check_sum += Parser.to_int(char) * pattern[idx % 3]  # Add the multiplication of the digit with the pattern

        return check_sum % 10 == int(digit)  # Check if the remainder matches

    @staticmethod
    def to_date(date: str) -> dict[str, str | None]:
        """
        Convert an MRZ date to a date object.
        :param date: A 6 character long MRZ date.
        :return: A date object.
        """
        return {
            'year': date[:2] if date[:2] != '<<' else '00',
            'month': date[2:4] if date[2:4] != '<<' else '00',
            'day': date[4:] if date[4:] != '<<' else '00'
        }

    @staticmethod
    def replace_name_spaces(first_line: str) -> str:
        """
        If there are more than 3 '<' in a row, the first line will continue with '<' until the end.
        Sometimes, the model predicts '<' as 'K', and this fixes a lot of those mistakes.
        :param first_line: The first line of an MRZ
        """
        first_triple_arrow = first_line[5:].find('<<<')  # get the index of the first triple arrow on the string
        if first_triple_arrow == -1:  # No triple arrow was found
            return first_line
        return first_line[:5] + first_line[5: 5 + first_triple_arrow] + '<' * len(first_line[5 + first_triple_arrow:])

    @staticmethod
    def fix_invalid_chars(first_line: str, second_line: str) -> tuple[str, str]:
        """
        In the MRZ specification, certain positions may only hold certain types of character.
        If a character does not match the type for its position, change it to the most likely character that does.
        An explanation for the way the characters were selected is available in prediction_mapping.py.

        :param first_line: The first line of the MRZ
        :param second_line: The second line of the MRZ
        :return: MRZ lines with no invalid characters.
        """
        char_types = {  # The character types, and their validation function
            "P": lambda c: c == 'P',  # A 'P' character.
            "a": lambda c: c in ALPHA,  # An alphabetic character
            "A": lambda c: c in ALPHA + ARROW,  # An alphabetic character or a '<'
            "d": lambda c: c in NUMERIC,  # A numeric character
            "D": lambda c: c in NUMERIC + ARROW,  # A numeric character or a '<'
            "*": lambda c: c in ALPHA + NUMERIC + ARROW,  # All
            "S": lambda c: c in '<MF',  # '<'/'F'/'M', for sex
        }
        # The character
        line_1_map = "PAaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        line_2_map = "*********dAAADDDDDDdSDDDDDDd**************dd"

        fixed_mrz = ""  # Initialize the fixed MRZ string

        # Map over both lines and line maps in the same loop:
        for char, char_type in zip(first_line + second_line, line_1_map + line_2_map):
            if char_types[char_type](char):  # Character matches the format
                fixed_mrz += char
            else:  # Character does not match format
                if char_type == 'P':
                    # The character HAS to be a 'P'
                    fixed_mrz += 'P'
                else:
                    # Get the most likely character for the position and type.
                    fixed_mrz += MAPPINGS[char][char_type]

        # Return the fixed MRZ broken into the two lines.
        return fixed_mrz[:CHARACTERS_AMOUNT], fixed_mrz[CHARACTERS_AMOUNT:]

    def _parse_name(self, name: str):
        """
        Parse the name field to first and last name
        :param name: the name field
        :return: a name object; {first: str, last: str}
        """
        full_name = name.split('<<', 1)  # The first and last name are split with a "<<".
        if len(full_name) == 2:  # Both parts are found
            last, first = full_name
        else:
            # A prediction error (a "<<" is missing).
            self._meta['should_check_name'] = True  # Mark it as wrong
            last = full_name[0]  # Make everything the last name.
            first = ''
        last = last.replace('<', ' ')  # Replace filler arrows with spaces
        last = last.strip()  # Remove all leading and trailing spaces
        first = first.replace('<', ' ')  # Replace filler arrows with spaces
        first = first.strip()  # Remove all leading and trailing spaces

        if name[-1] != '<':
            # If the last character is not '<', there is a possibility the name was truncated.
            # Mark it as should check.
            self._meta['should_check_name'] = True

        return {
            'first': first,
            'last': last
        }

    def _preprocessing(self):
        """
        Preprocess the MRZ string to minimize errors.
        """
        # Fix first row arrows.
        self._mrz_line_1 = self.replace_name_spaces(self._mrz_line_1)
        # Fix invalid characters in both rows.
        self._mrz_line_1, self._mrz_line_2 = self.fix_invalid_chars(self._mrz_line_1, self._mrz_line_2)

    def _parse(self):
        """
        The main parsing method. Parses the MRZ lines by character positions, as defined in the standard.

        Saves the data to self._data, and metadata to self._meta.
        """
        self._data['issuer_iso'] = self._mrz_line_1[2:5].replace('<', '')

        self._data['name'] = self._parse_name(self._mrz_line_1[5:])

        self._data['passport_number'] = self._mrz_line_2[:9].replace('<', '')
        self._data['is_passport_number_valid'] = self.check_digit(self._mrz_line_2[9], self._mrz_line_2[:9])
        self._meta['should_check_passport_number'] = not self._data['is_passport_number_valid']

        self._data['nationality_iso'] = self._mrz_line_2[10:13].replace('<', '')

        self._data['date_of_birth'] = self.to_date(self._mrz_line_2[13:19])
        self._data['is_date_of_birth_valid'] = self.check_digit(self._mrz_line_2[19], self._mrz_line_2[13:19])
        self._meta['should_check_date_of_birth'] = not self._data['is_date_of_birth_valid']

        self._data['sex'] = self._mrz_line_2[20] if self._mrz_line_2[20] in 'MF' else None

        self._data['date_of_expiry'] = self.to_date(self._mrz_line_2[21:27])
        self._data['is_date_of_expiry_valid'] = self.check_digit(self._mrz_line_2[27], self._mrz_line_2[21:27])
        self._meta['should_check_date_of_expiry'] = not self._data['is_date_of_expiry_valid']

        self._data['personal_number'] = self._mrz_line_2[28:42].replace('<', '')
        self._data['is_personal_number_valid'] = self.check_digit(self._mrz_line_2[42], self._mrz_line_2[28:42])
        self._meta['should_check_personal_number'] = not self._data['is_personal_number_valid']

        self._data['is_mrz_valid'] = self.check_digit(
            self._mrz_line_2[43],
            self._mrz_line_2[0:10] + self._mrz_line_2[13:20] + self._mrz_line_2[21:43]
        )
        self._meta['should_check_mrz'] = not self._data['is_mrz_valid']
