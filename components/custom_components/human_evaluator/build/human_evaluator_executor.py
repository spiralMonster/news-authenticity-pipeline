import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv

import smtplib
import imaplib
import email
from email.message import EmailMessage
from email.utils import formatdate
from email.utils import make_msgid
from email_reply_parser import EmailReplyParser

from tfx.components.base.base_executor import BaseExecutor
from tfx.types.standard_artifacts import Artifact
from tfx.types import artifact_utils

import tensorflow as tf
import tensorflow_model_analysis as tfma

from typing_extensions import Dict,List,Text,Any

load_dotenv()


class HumanEvaluatorExecutor(BaseExecutor):
    """
    Executor for custom Human Evaluation (through email) Component.
    """

    def SendEmail(
            self,
            receiver_email_id: str,
            evaluation_file_path: str
    ):
        """
        Send email consisting of the details of the pipeline run to the Human Evaluator.
        """

        def generate_email_subject():
            """
            Generating email subject.
            """

            time = datetime.now()
            subject = f"[{time}] News Authentication Pipeline Evaluation"

            return subject

        def generate_email_content(evaluation_file_path: str):
            """
            Generating the content of the email
            """

            # Loading configs:
            config_folder = "configs/model_configs/"

            model_details_path = os.path.join(config_folder, "model_details.json")
            text_model_config_path = os.path.join(config_folder, "text_model_config.json")
            example_split_path = os.path.join(config_folder, "number_of_examples.json")
            training_config_path = os.path.join(config_folder, "training_configs.json")
            training_details_path = os.path.join(config_folder, "training_details.json")
            optimizer_config_path = os.path.join(config_folder, "optimizer_config.json")

            with open(model_details_path, "r") as file:
                model_details = json.load(file)

            with open(text_model_config_path, "r") as file:
                text_model_config = json.load(file)

            with open(example_split_path, "r") as file:
                example_split = json.load(file)

            with open(training_config_path, "r") as file:
                training_config = json.load(file)

            with open(training_details_path, "r") as file:
                training_details = json.load(file)

            with open(optimizer_config_path, "r") as file:
                optimizer_config = json.load(file)

            eval_result = tfma.load_eval_result(evaluation_file_path).slicing_metrics[0][1][""][""]
            test_binary_acc = round(eval_result["binary_accuracy"]["doubleValue"], 2)
            test_precision = round(eval_result["precision"]["doubleValue"], 2)
            test_recall = round(eval_result["recall"]["doubleValue"], 2)
            test_auc = round(eval_result["auc"]["doubleValue"], 2)

            content = f"""
            Hi Developer,

            The news authentication model has completed the training successfully.
            Given follow are some details about the pipeline run:

            Overview of the News Authentication Model:
             - Number of Parameters: {model_details['num_of_parameters']}
             - Model Size : {model_details['model_size']}


            Overview of the Text Processing Model:
             - Num of parallel text processing model: {text_model_config['num_models']}
             - Vocab Size: {text_model_config['vocab_size']}
             - Embedding Dimension: {text_model_config['embedding_dim']}
             - Text Sequence Length: {text_model_config['text_seq_len']}


            Number of examples in each Split:
             - Number of Training data examples: {example_split['number_of_examples_in_Split-train_data']}
             - Number of Validation data examples: {example_split['number_of_examples_in_Split-eval_data']}
             - Number of Testing data examples: {example_split['number_of_examples_in_Split-test_data']}


            Training Hyperparameters of News Authentication Model:
             - Batch Size: {training_config['BATCH_SIZE']}
             - Number of Training Steps: {training_config['NUM_TRAIN_STEPS']}
             - Number of Validation Steps: {training_config['NUM_EVAL_STEPS']}
             - Optimizer: Adam
             - Learning Rate: {optimizer_config['learning_rate']}
             - Loss Function: Categorical Cross Entropy


            Training Performance:
             - Average Training Loss: {training_details['avg_training_loss']}
             - Average Training Accuracy: {training_details['avg_training_accuracy']}


            Validation Performance:
             - Average Validation Loss: {training_details['avg_validation_loss']}
             - Average Validation Accuracy: {training_details['avg_validation_accuracy']}


            Testing Performance:
             - Binary Accuracy: {test_binary_acc}
             - Precision: {test_precision}
             - Recall: {test_recall}
             - AUC: {test_auc}


            For more details, visit the following jupyter notebook:
            'https://github.com/spiralMonster/news-authenticity-pipeline/blob/main/pipeline_components_notebook.ipynb'

            Reply with 'Approve' to continue the pipeline run.
            Reply with 'Disapprove' to stop the pipeline run.

            Regards,
            Human Evaluator Component
            """

            return content

        def send(
                receiver_email_id: str,
                email_subject: str,
                email_content: str

        ):
            """
            send the email.
            """

            EMAIL_ADDRESS = os.environ["EMAIL_ADDRESS"]
            APP_PASSWORD = os.environ["APP_PASSWORD"]

            email_id = make_msgid()

            msg = EmailMessage()
            msg["Subject"] = email_subject
            msg["From"] = EMAIL_ADDRESS
            msg["To"] = receiver_email_id
            msg["Date"] = formatdate(localtime=True)
            msg["Message-ID"] = email_id
            msg["Reply-To"] = EMAIL_ADDRESS

            msg.set_content(email_content)

            try:
                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp_cli:
                    smtp_cli.login(EMAIL_ADDRESS, APP_PASSWORD)
                    smtp_cli.send_message(msg)

                email_id = email_id.strip()
                return email_id

            except Exception as e:
                print(f"Exception encountered: {e}")
                return None

        print(f"[INFO] Generating email subject.")
        email_subject = generate_email_subject()

        print(f"[INFO] Generating email content.")
        email_content = generate_email_content(evaluation_file_path=evaluation_file_path)

        email_id = send(
            receiver_email_id=receiver_email_id,
            email_subject=email_subject,
            email_content=email_content
        )

        if email_id is None:
            return (email_id, None)

        else:
            return (email_id, email_subject)

    def ReadEmailReply(
            self,
            sent_email_id,
            email_subject
    ):
        """
        Read the Reply of the Human Evaluator
        """

        def get_email_body(msg):
            """
            Get the body of the email.
            """

            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body = payload.decode(errors="ignore")
                            return body

            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode(errors="ignore")
                    return body

            return None

        EMAIL_ADDRESS = os.environ["EMAIL_ADDRESS"]
        APP_PASSWORD = os.environ["APP_PASSWORD"]

        with imaplib.IMAP4_SSL("imap.gmail.com") as imap_cli:
            imap_cli.login(EMAIL_ADDRESS, APP_PASSWORD)
            imap_cli.select("INBOX")

            status, messages = imap_cli.search(
                None,
                f'(SUBJECT "{email_subject}")'
            )
            email_ids = messages[0].split()

            if len(email_ids) == 0:
                return None

            else:
                email_id = email_ids[-1]

                status, msg_data = imap_cli.fetch(email_id, "(RFC822)")
                msg = email.message_from_bytes(msg_data[0][1])

                in_reply_to = msg.get("In-Reply-To")
                if (in_reply_to is not None and sent_email_id == in_reply_to.strip()):

                    body = get_email_body(msg)
                    if body is None:
                        print(f"[Warning] Could not extract email body.")
                        return None

                    reply = EmailReplyParser.parse_reply(body).strip()
                    return reply


                else:
                    return None

    def Do(
            self,
            input_dict: Dict[Text, List[Artifact]],
            output_dict: Dict[Text, List[Artifact]],
            exec_properties: Dict[Text, Any]
    ):
        print(f"[START] Human Evaluator Component.")
        self._log_startup(input_dict, output_dict, exec_properties)

        output_blessing = artifact_utils.get_single_instance(
            output_dict["human_evaluator_blessing"]
        )
        input_blessing = artifact_utils.get_single_instance(
            input_dict["model_blessing"]
        )

        is_blessed = None

        output_dir = output_dict["human_evaluator_blessing"][-1].uri
        tf.io.gfile.makedirs(output_dir)

        evaluator_blessing_dir = input_dict["model_blessing"][-1].uri
        evaluator_blessing = os.listdir(evaluator_blessing_dir)[-1]

        if evaluator_blessing == "NOT_BLESSED":
            blessing = "NOT_BLESSED"
            is_blessed = False

            output = os.path.join(output_dir, blessing)

            with tf.io.gfile.GFile(output, "w") as file:
                file.write("")



        else:
            evaluation_file_path = input_dict["model_evaluation"][-1].uri

            receiver_email_id = exec_properties["human_evaluator_email_id"]
            num_retries = exec_properties["num_retries"]
            waiting_time = exec_properties["waiting_time"]

            email_sent = False

            print(f"[INFO] Sending Email")
            for ind in range(num_retries):
                sent_email_id, sent_email_subject = self.SendEmail(
                    receiver_email_id=receiver_email_id,
                    evaluation_file_path=evaluation_file_path
                )

                if sent_email_id is None:
                    print(f"[WARNING] Failed to send email.")
                    print(f"[INFO] Retrying sending email.")


                else:
                    print(f"[INFO] Email send successfully.")
                    email_sent = True
                    break

            if not email_sent:
                blessing = "NOT_BLESSED"
                is_blessed = False

                output = os.path.join(output_dir, blessing)

                with tf.io.gfile.GFile(output, "w") as file:
                    file.write("")



            else:
                reply_received = False

                print(f"[INFO] Reading Human Evaluator Response...")
                for _ in range(num_retries):
                    time.sleep(waiting_time)

                    reply = self.ReadEmailReply(
                        sent_email_id=sent_email_id,
                        email_subject=sent_email_subject
                    )

                    if reply is None:
                        print(f"[WARNING] No reply received yet.")
                        print(f"[INFO] Retrying after some time.")

                    else:
                        print(f"[INFO] The Reply Recieved.")
                        print(f"[INFO] Reply: {reply}")

                        reply_received = True
                        break

                if not reply_received:
                    blessing = "NOT_BLESSED"
                    is_blessed = False

                    output = os.path.join(output_dir, blessing)

                    with tf.io.gfile.GFile(output, "w") as file:
                        file.write("")


                else:
                    if reply == "Approve":
                        blessing = "BLESSED"
                        is_blessed = True

                    else:
                        blessing = "NOT_BLESSED"
                        is_blessed = False

                    output = os.path.join(output_dir, blessing)

                    with tf.io.gfile.GFile(output, "w") as file:
                        file.write("")

        output_blessing.set_int_custom_property(
            "current_model_id",
            input_blessing.get_int_custom_property("current_model_id")
        )

        output_blessing.set_string_custom_property(
            "current_model",
            input_blessing.get_string_custom_property("current_model")
        )

        output_blessing.set_int_custom_property(
            "blessed",
            1 if is_blessed else 0
        )



