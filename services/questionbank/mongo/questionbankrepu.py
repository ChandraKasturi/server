import pymongo

def insertIntoQuestionsU(cursorU,collectionU):
    idU = cursorU.sahasra_questions.question_bank.insert_one(dict(collectionU))
    return str(idU.inserted_id)

def CheckifEmailPresent(cursorU,collectionU,mobileno=None):
    if mobileno:
        idu = cursorU.sahasra_users.users.find_one({"$or":[{"email":{"$eq":f"{collectionU.lower()}"}},{"mobilenumber":{"$eq":f"{mobileno.lower()}"}}]})
    else:
        idu = cursorU.sahasra_users.users.find_one({"$or":[{"email":{"$eq":f"{collectionU.lower()}"}},{"mobilenumber":{"$eq":f"{collectionU.lower()}"}}]})
    print(collectionU)
    
    if idu:
        print(idu)
        return str(idu["email"])
    return False
