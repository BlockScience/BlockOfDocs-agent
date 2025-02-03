def get_user_name(slack_client, user_id):
    user_info = slack_client.users_info(user=user_id)
    return user_info['user']['name'], user_info['user']['profile']['display_name']
