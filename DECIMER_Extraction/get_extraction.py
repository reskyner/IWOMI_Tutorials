from openai import OpenAI
import json
import re

system_message = """Extract the compound information from the Paragraph."""


def create_user_message(row):
    return f"""{row['Main Text']}"""


def create_assistant_message(row):
    return f"""{{\"compound_group\":\"{row['compound_group']}\",\"compound_class\":\"{row['compound_class']}\",\"organism_part\":\"{row['organism_part']}\",\"organism_or_species\":\"{row['organism_or_species']}\",\"geo_location\":\"{row['geo_location']}\",\"Kingdom\":\"{row['Kingdom']}\",\"trivial_name\":\"{row['trivial_name']}\",\"location\":\"{row['location']}\",\"iupac_name\":\"{row['iupac_name']}\",\"abbreviation\":\"{row['abbreviation']}\",\"iupac_like_name\":\"{row['iupac_like_name']}\"}}"""


def prepare_example_conversation(row):
    messages = []
    messages.append({"role": "system", "content": system_message})

    user_message = create_user_message(row)
    messages.append({"role": "user", "content": user_message})

    assistant_message = create_assistant_message(row)
    messages.append({"role": "assistant", "content": assistant_message})

    return {"messages": messages}


client = OpenAI(api_key="")  # replace with your openai-api-key

retrieve_job_response = client.fine_tuning.jobs.retrieve(
    "ftjob-qxXVH583p3sAvCjWoYNxboQl"
)
# print(retrieve_job_response.model_dump_json(indent=2))
# print("status:", retrieve_job_response.status)
fine_tuned_model = retrieve_job_response.fine_tuned_model


def get_response(user_message):
    system_message = """Extract the compound information from the Paragraph."""
    test_messages = []
    test_messages.append({"role": "system", "content": system_message})
    test_messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=fine_tuned_model, messages=test_messages, temperature=0.3
    )

    return response.choices[0].message.content


def find_positions(main_text, extracted_text):
    positions = []
    labels = list(extracted_text.keys())
    for label in labels:
        values = extracted_text[label].split(", ")
        values = [
            x.strip("' ") for x in values
        ]  # Clean string from tailing and trailing whitespaces and apostrphes
        for value in values:
            if value != "nan":
                matches = re.finditer(
                    re.escape(value), main_text
                )  # Use finditer instead of search to find all mentions of that entity
                if matches:
                    for match in matches:
                        positions.append(
                            {
                                "label": label,
                                "start_offset": match.start(),
                                "end_offset": match.end(),  # removed value from position list since doccano only accepts label, start and end offset
                            }
                        )

                # removed the else condition here to avoid getting "nan" into the result list (positions)

    return positions


def get_spans(user_message):
    extracted_text = get_response(user_message)
    data_dict = json.loads(extracted_text)
    positions = find_positions(user_message, data_dict)

    return extracted_text, data_dict, positions
